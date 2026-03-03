import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast 
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score 
import os
import time
import requests 
from dotenv import load_dotenv
from tqdm import tqdm 
import torch.nn.functional as F  # ADDED for softmax
import gc

from dataset_factory import HAM10000Dataset, get_transforms
from model_factory import get_model

# ==========================================
#            FOCAL LOSS IMPLEMENTATION
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        # Compute cross entropy WITH label smoothing (prevents overconfident logits)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        
        # Compute p_t from UNSMOOTHED CE for accurate focal weighting
        ce_loss_hard = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss_hard)
        
        # Compute focal term: (1 - p_t)^gamma
        # This down-weights easy examples and focuses on hard ones
        focal_term = (1 - p_t) ** self.gamma
        
        # Apply class weights (alpha) if provided
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
        else:
            return focal_loss

# ==========================================
#               CONFIGURATION
# ==========================================
load_dotenv()
WEBHOOK_URL = os.getenv("WEB_HOOK") 

# MODEL_NAME = "swinv2_large_window12to24_192to384_22kft1k" # Model A
MODEL_NAME = "convnext_xlarge_384_in22ft1k"    # Model B
IMG_SIZE = 384

if "convnext" in MODEL_NAME:
    BATCH_SIZE = 64
else:    
    BATCH_SIZE = 24 

EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7 

# Layer-wise learning rates (lower for pretrained backbone, higher for head)
BACKBONE_LR = 2e-5
HEAD_LR = 1e-4
WARMUP_EPOCHS = 3  # Linear warmup before cosine annealing
NUM_WORKERS = 16
DATA_DIR = "./data"
IMG_DIR = "./data/all_images"
CHECKPOINT_DIR = "./checkpoints"
N_FOLDS = 5 

# Focal Loss hyperparameters
FOCAL_GAMMA = 2.0  # Focusing parameter (2.0 is standard)
LABEL_SMOOTHING = 0.1  # Prevents overconfident logits, reduces val_loss divergence
# NOTE: No alpha/class weights needed — WeightedRandomSampler handles class balance

# FiLM Conditioning (metadata fusion)
USE_FILM = True  # Enable FiLM conditioning with patient metadata (age, sex, localization)

def send_curl_log(data):
    if not WEBHOOK_URL or "YOUR_KEY_HERE" in WEBHOOK_URL:
        return 
    try:
        requests.post(WEBHOOK_URL, json=data, timeout=2)
    except Exception:
        pass

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available!")
        return
    
    print("="*50)
    print(f"Start training")
    print(f"Model: {MODEL_NAME}")
    print(f"Loss: Focal Loss (gamma={FOCAL_GAMMA})")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Batch Size: {BATCH_SIZE} ({BATCH_SIZE // torch.cuda.device_count()} per GPU)")
    print(f"Early Stopping: {EARLY_STOPPING_PATIENCE} epochs") # ADDED
    print("="*50)
    
    send_curl_log({"content": f"Starting 5-Fold CV: {MODEL_NAME} with Focal Loss (gamma={FOCAL_GAMMA}). Max Epochs: {EPOCHS}"})

    # Load the Master Folds CSV
    folds_path = os.path.join(DATA_DIR, "train_folds.csv")
    if not os.path.exists(folds_path):
        raise FileNotFoundError("Run '01_split_data.py' first!")
    
    full_df = pd.read_csv(folds_path)
    print(f"\nLoaded {len(full_df)} images for CV\n")
    
    # ==========================================
    #           START CROSS-VALIDATION
    # ==========================================
    fold_results = []  # Track results across folds
    
    for fold in range(N_FOLDS):
        print(f"\n{'#'*50}")
        print(f"   FOLD {fold + 1}/{N_FOLDS}")
        print(f"{'#'*50}")
        
        send_curl_log({"content": f"Starting Fold {fold + 1}/{N_FOLDS}"})

        # Dynamic Split
        train_df = full_df[full_df['fold'] != fold].reset_index(drop=True)
        val_df = full_df[full_df['fold'] == fold].reset_index(drop=True)
        
        print(f"Train: {len(train_df)} images | Val: {len(val_df)} images")

        # Datasets & Loaders
        transforms = get_transforms(IMG_SIZE)
        train_ds = HAM10000Dataset(train_df, IMG_DIR, transform=transforms['train'])
        val_ds = HAM10000Dataset(val_df, IMG_DIR, transform=transforms['val'])
        
        # WeightedRandomSampler: oversample minority classes so the model sees them more often
        label_map_sampler = train_ds.label_map
        train_labels = [label_map_sampler[row['dx']] for _, row in train_df.iterrows()]
        class_counts_sampler = np.bincount(train_labels, minlength=7)
        sample_weights = [1.0 / class_counts_sampler[lbl] for lbl in train_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
        
        # === AUGMENTATION SUMMARY ===
        idx_to_class = {v: k for k, v in label_map_sampler.items()}
        SHORT_NAMES = {0: 'Nevi', 1: 'Melanoma', 2: 'BKL', 3: 'BCC', 4: 'AKIEC', 5: 'Vascular', 6: 'Dermatofibroma'}
        total_samples_per_epoch = len(train_ds)  # num_samples for sampler
        print(f"\n{'='*60}")
        print(f"AUGMENTATION & SAMPLING SUMMARY (Fold {fold + 1})")
        print(f"{'='*60}")
        print(f"Original training set: {len(train_df)} unique lesions")
        print(f"Samples per epoch (with oversampling): {total_samples_per_epoch}")
        print(f"Augmentation: Every sample is randomly augmented on-the-fly")
        print(f"  -> Each epoch sees {total_samples_per_epoch} UNIQUE augmented views")
        print(f"\nOriginal class distribution:")
        for cls_idx in range(7):
            name = SHORT_NAMES[cls_idx]
            count = class_counts_sampler[cls_idx]
            pct = 100.0 * count / len(train_df)
            print(f"  [{cls_idx}] {name:15s}: {count:5d} ({pct:5.1f}%)")
        # Expected samples per class under WeightedRandomSampler (uniform)
        expected_per_class = total_samples_per_epoch // 7
        print(f"\nExpected samples per class per epoch (balanced sampler): ~{expected_per_class}")
        print(f"{'='*60}\n")

        train_loader = DataLoader(
            train_ds, 
            batch_size=BATCH_SIZE, 
            sampler=sampler,  # Replaces shuffle=True
            num_workers=NUM_WORKERS, 
            pin_memory=True, 
            persistent_workers=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS, 
            pin_memory=True,
            persistent_workers=True
        )

        # NOTE: Class weights are NOT used in Focal Loss because the WeightedRandomSampler
        # already balances class frequencies in every batch. Using both would double-compensate,
        # causing massive over-penalization of minority classes (~3600x for Dermatofibroma).
        # Focal Loss gamma still handles hard-example focusing without class weights.

        # Initialize Model (FRESH for every fold)
        model = get_model(MODEL_NAME, num_classes=7, use_film=USE_FILM)
        
        # Layer-wise LR: separate backbone, FiLM, and head parameters
        head_params = []
        film_params = []
        backbone_params = []
        for name, param in model.named_parameters():
            if 'film_generator' in name:
                film_params.append(param)
            elif 'head' in name or 'classifier' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        print(f"Parameters: {len(backbone_params)} backbone, {len(film_params)} FiLM, {len(head_params)} head")
        print(f"LR: backbone={BACKBONE_LR}, FiLM/head={HEAD_LR}")
        
        if torch.cuda.device_count() > 1:
            print(f"Wrapping with DataParallel ({torch.cuda.device_count()} GPUs)\n")
            model = nn.DataParallel(model)
        model = model.to(device)

        # Optimizer, Scheduler, Scaler
        # Focal Loss with gamma only (no alpha) — sampler handles class balance
        criterion = FocalLoss(alpha=None, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
        print(f"Using Focal Loss with gamma={FOCAL_GAMMA}, label_smoothing={LABEL_SMOOTHING}\n")
        
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': BACKBONE_LR},
            {'params': film_params, 'lr': HEAD_LR},
            {'params': head_params, 'lr': HEAD_LR}
        ], weight_decay=0.05)
        
        # Cosine Annealing with Linear Warmup
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-7)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])
        scaler = GradScaler('cuda') 

        # History
        history = {
            'epoch': [], 
            'train_loss': [], 
            'val_loss': [], 
            'val_acc': [], 
            'val_f1': [],
            'val_precision': [], 
            'val_recall': [] 
        }
        best_fold_f1 = 0.0
        start_epoch = 0
        epochs_no_improve = 0 # ADDED: Tracker for early stopping

        # ==========================================
        #           RESUME LOGIC
        # ==========================================
        resume_path = os.path.join(CHECKPOINT_DIR, f"last_checkpoint_fold{fold}.pth")
        log_path = os.path.join(CHECKPOINT_DIR, f"log_fold{fold}.csv")

        if os.path.exists(resume_path):
            print(f"Found checkpoint for Fold {fold}. Resuming...")
            checkpoint = torch.load(resume_path, map_location=device)
            
            # Load states
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
                
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # Safely load scheduler state (may fail if scheduler type changed)
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception:
                print("[WARNING] Scheduler state mismatch (likely changed type), reinitializing scheduler.")
            
            start_epoch = checkpoint['epoch']
            best_fold_f1 = checkpoint['best_f1']
            
            # ADDED: Safely load early stopping counter if it exists in old checkpoint
            epochs_no_improve = checkpoint.get('epochs_no_improve', 0) 
            
            print(f"Resuming from epoch {start_epoch + 1}/{EPOCHS}")
            print(f"Best F1 so far: {best_fold_f1:.4f}\n")
            
            # Reload history (with column validation)
            if os.path.exists(log_path):
                loaded_history = pd.read_csv(log_path).to_dict(orient='list')
                if set(loaded_history.keys()) == set(history.keys()):
                    history = loaded_history
                else:
                    print(f"[WARNING] History CSV columns mismatch, starting fresh history.")

        # Check if already done
        # A fold is complete if: no resume checkpoint AND the best model file exists
        best_model_path = os.path.join(CHECKPOINT_DIR, f"best_{MODEL_NAME}_fold{fold}.pth")
        if start_epoch >= EPOCHS or (not os.path.exists(resume_path) and os.path.exists(best_model_path)):
            print(f"Fold {fold + 1} already complete! Skipping.\n")
            if os.path.exists(log_path):
                fold_df = pd.read_csv(log_path)
                fold_results.append(fold_df['val_f1'].max())
            else:
                # Fallback: read best F1 from saved checkpoint
                ckpt = torch.load(best_model_path, map_location='cpu')
                fold_results.append(ckpt.get('best_f1', 0.0))
            continue

        # ==========================
        #       TRAINING LOOP
        # ==========================
        for epoch in range(start_epoch, EPOCHS):
            start_time = time.time()
            model.train()
            running_loss = 0.0
            epoch_class_counts = np.zeros(7, dtype=int)  # Track per-class samples this epoch
            current_lr = optimizer.param_groups[0]['lr']
            
            train_loop = tqdm(
                train_loader, 
                desc=f"Fold {fold + 1} Epoch {epoch + 1}/{EPOCHS}", 
                leave=False
            )

            for i, (images, metadata, labels) in enumerate(train_loop):
                images = images.to(device, non_blocking=True)
                metadata = metadata.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Count class distribution for this batch
                batch_counts = np.bincount(labels.cpu().numpy(), minlength=7)
                epoch_class_counts += batch_counts
                
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
                train_loop.set_postfix(loss=loss.item(), lr=f"{current_lr:.2e}")
                
                # Memory check (First epoch only)
                if fold == 0 and epoch == 0 and i == 1:
                    tqdm.write(f"\n{'='*50}")
                    tqdm.write("[GPU Memory Check]")
                    for gpu_id in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                        tqdm.write(f"  GPU {gpu_id}: {allocated:.2f}GB / 24GB (reserved: {reserved:.2f}GB)")
                        if allocated > 20:
                            tqdm.write(f"WARNING: Memory usage is high!")
                    tqdm.write(f"{'*'*50}\n")

            epoch_loss = running_loss / len(train_loader)

            # Print per-epoch class distribution
            total_seen = epoch_class_counts.sum()
            class_dist_str = " | ".join([f"{SHORT_NAMES[c]}: {epoch_class_counts[c]}" for c in range(7)])
            print(f"  Epoch {epoch + 1} samples seen: {total_seen} -> {class_dist_str}")

            # Save class distribution to CSV
            dist_log_path = os.path.join(CHECKPOINT_DIR, f"class_distribution_fold{fold}.csv")
            dist_row = {'epoch': epoch + 1, 'total_samples': int(total_seen)}
            for c in range(7):
                dist_row[SHORT_NAMES[c]] = int(epoch_class_counts[c])
            dist_df_row = pd.DataFrame([dist_row])
            if os.path.exists(dist_log_path):
                dist_df_row.to_csv(dist_log_path, mode='a', header=False, index=False)
            else:
                dist_df_row.to_csv(dist_log_path, index=False)

            # Validation
            model.eval()
            val_running_loss = 0.0
            all_preds = []
            all_labels = []
            all_probs = []
            
            val_loop = tqdm(val_loader, desc="Validation", leave=False)

            with torch.no_grad():
                for images, metadata, labels in val_loop:
                    images = images.to(device, non_blocking=True)
                    metadata = metadata.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    with autocast(device_type='cuda'):
                        outputs = model(images, metadata)
                        v_loss = criterion(outputs, labels)
                    
                    val_running_loss += v_loss.item()
                    
                    # Store Raw Probabilities for ROC
                    probs = F.softmax(outputs, dim=1)
                    all_probs.extend(probs.cpu().numpy())
                    
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            val_loss = val_running_loss / len(val_loader)
            val_acc = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, average='macro')
            
            val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            
            scheduler.step()

            # Logging
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(epoch_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            history['val_precision'].append(val_precision) 
            history['val_recall'].append(val_recall)       
            
            # SAVES CSV FOR GRAPHING LATER
            pd.DataFrame(history).to_csv(log_path, index=False)
            
            duration = time.time() - start_time
            print(f"Epoch {epoch + 1}/{EPOCHS} | {duration:.1f}s | "
                  f"Train Loss: {epoch_loss:.4f} | Val F1: {val_f1:.4f} | "
                  f"Prec: {val_precision:.4f} | Rec: {val_recall:.4f}")
            
            # Discord tracking 
            if (epoch + 1) % 1 == 0 or (epoch + 1) == EPOCHS:  
                send_curl_log({
                    "content": f"**Fold {fold + 1} - Epoch {epoch + 1}**\n"
                               f"F1: {val_f1:.4f} | Prec: {val_precision:.4f} | Rec: {val_recall:.4f} | Loss: {epoch_loss:.4f}"
                })

            # Save best model & PREDICTIONS FOR ROC
            if val_f1 > best_fold_f1:
                best_fold_f1 = val_f1
                epochs_no_improve = 0  # ADDED: Reset the counter because we found a new best!
                save_name = f"best_{MODEL_NAME}_fold{fold}.pth"
                save_path = os.path.join(CHECKPOINT_DIR, save_name)
                
                model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                
                checkpoint = {
                    'fold': fold,
                    'epoch': epoch + 1,
                    'model_state_dict': model_state,
                    'best_f1': best_fold_f1,
                    'val_acc': val_acc,
                    'val_precision': val_precision,
                    'val_recall': val_recall,
                    'config': {'model': MODEL_NAME, 'img_size': IMG_SIZE, 'loss': 'FocalLoss', 'gamma': FOCAL_GAMMA, 'use_film': USE_FILM}
                }
                torch.save(checkpoint, save_path)
                print(f"   Saved best model: {save_name} (F1: {best_fold_f1:.4f})")
                
                # NEW: Send webhook for new best
                send_curl_log({
                    "content": f"**NEW BEST - Fold {fold + 1}**\nEpoch {epoch + 1} | F1: {best_fold_f1:.4f}"
                })
                
                # === NEW: SAVE PREDICTIONS FOR ROC CURVE ===
                # We save a CSV with: True Label, Prob_Class_0, Prob_Class_1, ...
                preds_df = pd.DataFrame(all_probs, columns=[f"prob_{i}" for i in range(7)])
                preds_df['true_label'] = all_labels
                preds_path = os.path.join(CHECKPOINT_DIR, f"predictions_fold{fold}.csv")
                preds_df.to_csv(preds_path, index=False)
                print(f"   Saved ROC predictions to {preds_path}")
            else:
                epochs_no_improve += 1  # ADDED: Increment the early stopping counter
                print(f"   Early stopping counter: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")

            # Save last checkpoint (for resume)
            last_checkpoint = {
                'fold': fold,
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_fold_f1,
                'epochs_no_improve': epochs_no_improve # ADDED: Save early stopping state
            }
            torch.save(last_checkpoint, resume_path)

            # ADDED: Break the loop if patience is exceeded
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\n[EARLY STOPPING] Fold {fold + 1} triggered early stopping! No improvement for {EARLY_STOPPING_PATIENCE} epochs.")
                send_curl_log({"content": f" **Early Stopping Triggered** for Fold {fold + 1} at Epoch {epoch + 1}."})
                break
        
        # Cleanup resume checkpoint after successful completion
        if os.path.exists(resume_path):
            os.remove(resume_path)
            print(f"Cleaned up resume checkpoint")
        
        # Free GPU memory between folds
        # Sync all GPUs to ensure async ops are complete
        for gpu_id in range(torch.cuda.device_count()):
            torch.cuda.synchronize(gpu_id)
        
        # Delete DataLoaders first to shut down persistent workers
        del train_loader, val_loader, train_ds, val_ds
        del model, optimizer, scheduler, scaler, criterion
        gc.collect()
        torch.cuda.empty_cache()
        
        # Give OS time to fully reclaim worker process memory
        time.sleep(5)
        print(f"GPU memory after cleanup:")
        for gpu_id in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            print(f"  GPU {gpu_id}: {allocated:.2f}GB allocated")
        
        # Track result
        fold_results.append(best_fold_f1)
        
        print(f"\n Fold {fold + 1} complete! Best F1: {best_fold_f1:.4f}")
        send_curl_log({"content": f" **Fold {fold + 1} Complete** | Best F1: {best_fold_f1:.4f}"})
        print("-" * 50)

    # Final summary
    print(f"\n{'='*50}")
    print("CROSS-VALIDATION COMPLETE")
    print(f"{'='*50}")
    
    if len(fold_results) == N_FOLDS:
        mean_f1 = np.mean(fold_results)
        std_f1 = np.std(fold_results)
        min_f1 = min(fold_results)
        max_f1 = max(fold_results)
        
        print(f"Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"Min F1:  {min_f1:.4f}")
        print(f"Max F1:  {max_f1:.4f}")
        print(f"Per-fold F1s: {[f'{f:.4f}' for f in fold_results]}")
        
        send_curl_log({
            "content": f" **ALL FOLDS COMPLETE (Focal Loss)**\n"
                      f"Mean F1: {mean_f1:.4f} ± {std_f1:.4f}\n"
                      f"Range: [{min_f1:.4f}, {max_f1:.4f}]"
        })
    else:
        print(f"Warning: Only {len(fold_results)}/{N_FOLDS} folds completed")
    
    print(f"{'='*50}")

if __name__ == "__main__":
    main()