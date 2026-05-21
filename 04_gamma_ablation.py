"""
Focal Loss γ Ablation on HAM10000
==================================
Tests γ ∈ {0.5, 1.0, 2.0, 3.0, 5.0} with label_smoothing=0.1 on a single fold
to find the optimal focusing parameter for our specific 7-class medical imbalance.

Motivation:
-----------
Lin et al. tuned γ on COCO (80 classes, natural images). HAM10000 has a fundamentally
different imbalance structure (7 classes, medical labels, offline-balanced via augmentation).
Additionally, label smoothing (ε=0.1) dampens focal loss's focusing effect:
  - Smoothing prevents overconfident predictions → p_t stays lower
  - Lower p_t → focal term (1-p_t)^γ is larger → less down-weighting of easy examples
  - Net effect: γ=2.0 under smoothing behaves roughly like γ≈1.7 without smoothing
  - The optimal γ under smoothing is likely HIGHER than the unsmoothed optimum

This ablation tests each γ on fold 0 with the same training setup as train.py,
then produces a comparison table of per-class and macro metrics.

Usage:
------
    python 04_gamma_ablation.py

Output:
-------
    ./ablation_gamma/results_summary.csv    — Comparison table
    ./ablation_gamma/gamma_X.X/             — Per-gamma checkpoints & logs
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report
)
import os
import time
import gc
from tqdm import tqdm

from dataset_factory import HAM10000Dataset, get_transforms
from model_factory import get_model

# ==========================================
#          ABLATION CONFIGURATION
# ==========================================
# Gamma values to test
GAMMA_VALUES = [0.5, 1.0, 2.0, 3.0, 5.0]

# Fixed hyperparameters (same as train.py)
LABEL_SMOOTHING = 0.1
MODEL_NAME = "convnext_xlarge_384_in22ft1k"
IMG_SIZE = 384
BATCH_SIZE = 64
ABLATION_EPOCHS = 20       # Reduced from 50 — enough to see convergence trends
EARLY_STOPPING_PATIENCE = 7
BACKBONE_LR = 2e-5
HEAD_LR = 1e-4
WARMUP_EPOCHS = 3
NUM_WORKERS = 16
DATA_DIR = "./data"
IMG_DIR = "./data/all_images"
AUG_DIR = "./data/augmented"
ABLATION_DIR = "./ablation_gamma"
USE_FILM = True
ABLATION_FOLD = 0  # Single fold for efficiency

CLASS_NAMES = {0: 'nv', 1: 'mel', 2: 'bkl', 3: 'bcc', 4: 'akiec', 5: 'vasc', 6: 'df'}

# ==========================================
#          FOCAL LOSS (from train.py)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none',
                                  label_smoothing=self.label_smoothing)
        # p_t from unsmoothed CE for accurate focal weighting
        ce_loss_hard = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss_hard)
        focal_term = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def run_single_gamma(gamma, fold, full_df, device):
    """Train one fold with a specific gamma value. Returns metrics dict."""
    
    gamma_dir = os.path.join(ABLATION_DIR, f"gamma_{gamma}")
    os.makedirs(gamma_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  ABLATION: γ = {gamma}  |  fold = {fold}")
    print(f"{'='*60}")

    # --- Data ---
    train_df = full_df[full_df['fold'] != fold].reset_index(drop=True)
    val_df = full_df[full_df['fold'] == fold].reset_index(drop=True)

    balanced_csv = os.path.join(DATA_DIR, f"balanced_fold{fold}.csv")
    if not os.path.exists(balanced_csv):
        raise FileNotFoundError(f"Run 02_balance_dataset.py first! Missing: {balanced_csv}")
    balanced_train_df = pd.read_csv(balanced_csv)
    fold_aug_dir = os.path.join(AUG_DIR, f"fold{fold}")

    transforms = get_transforms(IMG_SIZE)
    train_ds = HAM10000Dataset(balanced_train_df, IMG_DIR, transform=transforms['train'], aug_img_dir=fold_aug_dir)
    val_ds = HAM10000Dataset(val_df, IMG_DIR, transform=transforms['val'])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )

    # --- Model ---
    model = get_model(MODEL_NAME, num_classes=7, use_film=USE_FILM)

    head_params = []
    film_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if 'film_generator' in name or 'stage_films' in name:
            film_params.append(param)
        elif 'head' in name or 'classifier' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # --- Loss / Optimizer ---
    criterion = FocalLoss(alpha=None, gamma=gamma, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': BACKBONE_LR},
        {'params': film_params, 'lr': HEAD_LR},
        {'params': head_params, 'lr': HEAD_LR}
    ], weight_decay=0.05)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=ABLATION_EPOCHS - WARMUP_EPOCHS, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])
    scaler = GradScaler('cuda')

    # --- Training ---
    history = []
    best_f1 = 0.0
    best_metrics = {}
    epochs_no_improve = 0

    for epoch in range(ABLATION_EPOCHS):
        t0 = time.time()
        model.train()
        running_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"γ={gamma} Epoch {epoch+1}/{ABLATION_EPOCHS}", leave=False)
        for images, metadata, labels in train_loop:
            images = images.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda'):
                outputs = model(images, metadata)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_running_loss = 0.0
        all_preds, all_labels_list, all_probs = [], [], []

        with torch.no_grad():
            for images, metadata, labels in val_loader:
                images = images.to(device, non_blocking=True)
                metadata = metadata.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast(device_type='cuda'):
                    outputs = model(images, metadata)
                    v_loss = criterion(outputs, labels)

                val_running_loss += v_loss.item()
                probs = F.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels_list.extend(labels.cpu().numpy())

        val_loss = val_running_loss / len(val_loader)
        val_f1 = f1_score(all_labels_list, all_preds, average='macro')
        val_acc = accuracy_score(all_labels_list, all_preds)
        val_prec = precision_score(all_labels_list, all_preds, average='macro', zero_division=0)
        val_rec = recall_score(all_labels_list, all_preds, average='macro', zero_division=0)

        # Per-class F1
        per_class_f1 = f1_score(all_labels_list, all_preds, average=None)

        scheduler.step()
        duration = time.time() - t0

        row = {
            'epoch': epoch + 1, 'gamma': gamma, 'train_loss': epoch_loss,
            'val_loss': val_loss, 'val_f1': val_f1, 'val_acc': val_acc,
            'val_precision': val_prec, 'val_recall': val_rec,
        }
        for c in range(7):
            row[f'f1_{CLASS_NAMES[c]}'] = per_class_f1[c] if c < len(per_class_f1) else 0.0
        history.append(row)

        print(f"  Epoch {epoch+1} ({duration:.0f}s) | loss={epoch_loss:.4f} | "
              f"val_f1={val_f1:.4f} | acc={val_acc:.4f} | prec={val_prec:.4f} | rec={val_rec:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            best_metrics = {
                'gamma': gamma, 'best_epoch': epoch + 1,
                'val_f1': val_f1, 'val_acc': val_acc,
                'val_precision': val_prec, 'val_recall': val_rec,
                'val_loss': val_loss, 'train_loss': epoch_loss,
            }
            for c in range(7):
                best_metrics[f'f1_{CLASS_NAMES[c]}'] = per_class_f1[c] if c < len(per_class_f1) else 0.0

            # Save best model
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({
                'model_state_dict': model_state, 'gamma': gamma, 'best_f1': best_f1,
                'epoch': epoch + 1
            }, os.path.join(gamma_dir, f"best_model_fold{fold}.pth"))
            print(f"    ★ New best F1={best_f1:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    # Save training log
    pd.DataFrame(history).to_csv(os.path.join(gamma_dir, f"log_fold{fold}.csv"), index=False)

    # Save full classification report at best epoch
    print(f"\n  γ={gamma} BEST: F1={best_metrics['val_f1']:.4f} at epoch {best_metrics['best_epoch']}")

    # Cleanup
    for gpu_id in range(torch.cuda.device_count()):
        torch.cuda.synchronize(gpu_id)
    del train_loader, val_loader, train_ds, val_ds
    del model, optimizer, scheduler, scaler, criterion
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)

    return best_metrics


def main():
    os.makedirs(ABLATION_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available!")
        return

    print("=" * 60)
    print("FOCAL LOSS γ ABLATION STUDY")
    print(f"Model: {MODEL_NAME}")
    print(f"Label Smoothing: ε = {LABEL_SMOOTHING}")
    print(f"γ values: {GAMMA_VALUES}")
    print(f"Fold: {ABLATION_FOLD} | Max epochs: {ABLATION_EPOCHS}")
    print("=" * 60)

    # Quantify the smoothing-gamma interaction
    K = 7   # number of classes
    eps = LABEL_SMOOTHING
    p_smooth_max = 1.0 - eps * (K - 1) / K
    print(f"\n[ANALYSIS] Label smoothing interaction:")
    print(f"  With ε={eps}, K={K}: max smoothed target = {p_smooth_max:.4f}")
    print(f"  This means the network learns to cap confidence below ~{p_smooth_max:.1%}")
    print(f"  For a perfectly confident model (p_t=0.95):")
    for g in GAMMA_VALUES:
        focal_weight = (1 - 0.95) ** g
        focal_weight_damped = (1 - 0.90) ** g  # lower p_t due to smoothing
        print(f"    γ={g}: focal_weight = {focal_weight:.6f} "
              f"(undamped) vs {focal_weight_damped:.6f} (smoothed p_t≈0.90) "
              f"→ {focal_weight_damped/focal_weight:.1f}x more gradient")
    print()

    folds_path = os.path.join(DATA_DIR, "train_folds.csv")
    if not os.path.exists(folds_path):
        raise FileNotFoundError("Run '01_split_data.py' first!")
    full_df = pd.read_csv(folds_path)

    # Run ablation for each gamma
    all_results = []
    for gamma in GAMMA_VALUES:
        metrics = run_single_gamma(gamma, ABLATION_FOLD, full_df, device)
        all_results.append(metrics)

    # ==========================================
    #           RESULTS SUMMARY
    # ==========================================
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('val_f1', ascending=False).reset_index(drop=True)
    results_path = os.path.join(ABLATION_DIR, "results_summary.csv")
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 80)
    print("γ ABLATION RESULTS (sorted by macro F1)")
    print("=" * 80)

    # Display comparison table
    header = f"{'γ':>5} | {'F1':>7} | {'Acc':>7} | {'Prec':>7} | {'Rec':>7} | {'Epoch':>5} | "
    header += " | ".join([f"{CLASS_NAMES[c]:>5}" for c in range(7)])
    print(header)
    print("-" * len(header))

    for _, row in results_df.iterrows():
        line = f"{row['gamma']:5.1f} | {row['val_f1']:7.4f} | {row['val_acc']:7.4f} | "
        line += f"{row['val_precision']:7.4f} | {row['val_recall']:7.4f} | {int(row['best_epoch']):5d} | "
        line += " | ".join([f"{row[f'f1_{CLASS_NAMES[c]}']:5.3f}" for c in range(7)])
        print(line)

    best_gamma = results_df.iloc[0]['gamma']
    best_f1 = results_df.iloc[0]['val_f1']
    print(f"\n→ OPTIMAL γ = {best_gamma} (F1 = {best_f1:.4f})")

    # Recommendation
    current_gamma = 2.0
    if best_gamma != current_gamma:
        improvement = results_df.iloc[0]['val_f1'] - results_df[results_df['gamma'] == current_gamma]['val_f1'].values[0]
        print(f"  Δ vs current γ={current_gamma}: {improvement:+.4f} F1")
        print(f"  → UPDATE FOCAL_GAMMA in train.py to {best_gamma}")
    else:
        print(f"  Current γ={current_gamma} is already optimal.")

    # Check per-class impact (minority class sensitivity)
    print(f"\nPer-class F1 sensitivity to γ:")
    for c in range(7):
        col = f"f1_{CLASS_NAMES[c]}"
        best_g = results_df.loc[results_df[col].idxmax(), 'gamma']
        worst_g = results_df.loc[results_df[col].idxmin(), 'gamma']
        spread = results_df[col].max() - results_df[col].min()
        print(f"  {CLASS_NAMES[c]:>6}: best γ={best_g}, worst γ={worst_g}, spread={spread:.4f}")

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
