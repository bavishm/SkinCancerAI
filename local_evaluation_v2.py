import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, 
    classification_report, accuracy_score, f1_score
)
from torch.utils.data import DataLoader
from torch.amp import autocast 
from itertools import cycle
import os
import gc
from tqdm import tqdm

from dataset_factory import HAM10000Dataset, get_transforms
from model_factory import get_model
import re


def remap_swin_state_dict(state_dict, model_state_dict):
    """Remap state_dict keys for timm version compatibility.
    
    Older timm puts downsample at end of stage N (layers.N.downsample),
    newer timm puts it at start of stage N+1 (layers.N+1.downsample).
    Also strips non-persistent attn_mask buffers.
    """
    # Check if remapping is needed by looking for the version mismatch signature
    has_old_keys = any("layers.0.downsample" in k for k in state_dict)
    needs_new_keys = any("layers.3.downsample" in k for k in model_state_dict)
    
    if not (has_old_keys and needs_new_keys):
        return state_dict
    
    print("[INFO] Remapping state_dict keys for timm version compatibility...")
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        # Shift downsample layer indices: N → N+1
        match = re.match(r"(.*layers\.)(\d+)(\.downsample\..*)", key)
        if match:
            old_idx = int(match.group(2))
            new_key = f"{match.group(1)}{old_idx + 1}{match.group(3)}"
        # Skip non-persistent attn_mask buffers
        if "attn_mask" in key and key not in model_state_dict:
            continue
        new_state_dict[new_key] = value
    return new_state_dict

# ==========================================
#               CONFIGURATION
# ==========================================
MODEL_NAME = "swinv2_large_window12to24_192to384_22kft1k" # Model A
# MODEL_NAME = "convnext_xlarge_384_in22ft1k"    # Model B
IMG_SIZE = 384
BATCH_SIZE = 16   
NUM_WORKERS = 0    

TEST_CSV_PATH = "./data/test_split.csv" 
IMG_DIR = "./data/all_images"
CHECKPOINT_DIR = "./checkpoints_swinv2_focal_film_aug" # Model A
# CHECKPOINT_DIR = "./checkpoints_convnext_focal_film_aug"

# OUTPUT_DIR = "./eval_data/5_swinv2_focal_tta" # Model A
OUTPUT_DIR = "./eval_data/5_swinv2_focal_film_aug" # Model B
MELANOMA_THRESHOLD = 0.20
ENABLE_TTA = False  # Test-Time Augmentation (original + hflip + vflip + hflip+vflip)
USE_FILM = True  # Must match the training setting

# Per-class optimized weights (from 03_optimize_thresholds.py)
# Loaded from checkpoint dir so weights stay with their model
CLASS_WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "class_weights_optimized.npy")  # Set to None to disable

# Dataset class names 
CLASSES = [
    'Melanomaanocytic Nevi (moles)',                      #  0
    'Melanoma',                                           #  1
    'Benign Keratosis-like Lesions',                      #  2
    'Basal Cell Carcinoma',                               #  3
    'Actinic Keratoses & Intraepithelial Carcinoma',      #  4
    'Vascular Lesions',                                   #  5
    'Dermatofibroma'                                      #  6
]

def plot_confusion_matrix(y_true, y_pred, classes, save_path, title_prefix=""):
    """Plot and save normalized confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    short_labels = [c[:20] + '...' if len(c) > 20 else c for c in classes]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues', 
        xticklabels=short_labels, 
        yticklabels=short_labels,
        cbar_kws={'label': 'Proportion'}
    )
    plt.title(f'Normalized Confusion Matrix\n({title_prefix})', fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_multiclass_roc(y_true, y_probs, classes, save_path, title_prefix=""):
    """Plot and save multi-class ROC curves"""
    y_true_bin = pd.get_dummies(y_true).values
    n_classes = len(classes)
    
    fpr, tpr, roc_auc = {}, {}, {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(12, 8))
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'])
    
    for i, color in zip(range(n_classes), colors):
        label = classes[i][:25] + '...' if len(classes[i]) > 25 else classes[i]
        plt.plot(
            fpr[i], tpr[i], 
            color=color, 
            lw=2,
            label=f'{label} (AUC={roc_auc[i]:.3f})'
        )

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Multi-Class ROC Curves\n({title_prefix})', fontsize=14, pad=20)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def tta_inference(model, images, metadata, use_amp):
    """Test-Time Augmentation: run model on original + 3 flipped versions, average probs."""
    augmented = [
        images,                           # original
        torch.flip(images, dims=[3]),     # horizontal flip
        torch.flip(images, dims=[2]),     # vertical flip
        torch.flip(images, dims=[2, 3])   # both flips
    ]
    probs_sum = None
    for aug_images in augmented:
        if use_amp:
            with autocast(device_type='cuda'):
                outputs = model(aug_images, metadata)
        else:
            outputs = model(aug_images, metadata)
        probs = F.softmax(outputs, dim=1)
        if probs_sum is None:
            probs_sum = probs
        else:
            probs_sum += probs
    return (probs_sum / len(augmented)).cpu().numpy()

# ADDED: Reusable function to generate and save all metrics for a specific prediction set
def evaluate_and_save(y_true, y_probs, y_preds, out_dir, title_prefix, is_adjusted=False, override_stats=None):
    os.makedirs(out_dir, exist_ok=True)
    
    test_acc = accuracy_score(y_true, y_preds)
    test_f1_macro = f1_score(y_true, y_preds, average='macro')
    test_f1_weighted = f1_score(y_true, y_preds, average='weighted')
    report_str = classification_report(y_true, y_preds, target_names=CLASSES, digits=4)
    
    # Text Report
    output_text = f"""{"="*70}\n {title_prefix.upper()}\n{"="*70}
Model:               {MODEL_NAME}
"""
    if is_adjusted and override_stats:
        output_text += f"""
CLINICAL TRIAGE OVERRIDE (Threshold: {MELANOMA_THRESHOLD})
  Additional melanomas flagged: {override_stats['num_overrides']}
  Original melanoma predictions: {override_stats['old_count']}
  After threshold adjustment:    {override_stats['new_count']}
"""
    
    output_text += f"""
Overall Metrics:
  Accuracy:           {test_acc:.4f}
  F1 Score (Macro):   {test_f1_macro:.4f}
  F1 Score (Weighted): {test_f1_weighted:.4f}

{"="*70}
 PER-CLASS CLASSIFICATION REPORT
{"="*70}
{report_str}
"""
    # Save Report
    with open(os.path.join(out_dir, 'test_results.txt'), 'w') as f:
        f.write(output_text)
        
    # Generate Plots
    plot_confusion_matrix(y_true, y_preds, CLASSES, os.path.join(out_dir, 'confusion_matrix.png'), title_prefix)
    plot_multiclass_roc(y_true, y_probs, CLASSES, os.path.join(out_dir, 'roc_curves.png'), title_prefix)
    
    # Save CSV with raw probabilities
    cols = [f"prob_{i}" for i in range(len(CLASSES))]
    df = pd.DataFrame(y_probs, columns=cols)
    df['true_label'] = y_true
    df['prediction'] = y_preds
    df.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)
    
    # Save NPY
    results = {
        'predictions': y_preds,
        'true_labels': y_true,
        'probabilities': y_probs,
        'class_names': CLASSES,
        'metrics': {
            'accuracy': float(test_acc),
            'f1_macro': float(test_f1_macro),
            'f1_weighted': float(test_f1_weighted)
        },
        'model_name': MODEL_NAME
    }
    np.save(os.path.join(out_dir, 'test_results.npy'), results)
    print(f"  [+] Saved {title_prefix} results to {out_dir}/")

def main():
    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        use_amp = False
        print("Using CPU")
    
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Sequential Inference: Enabled (loads 1 model at a time)")
    print(f"  Melanoma Threshold: {MELANOMA_THRESHOLD}")
    print(f"  Test-Time Augmentation: {'ENABLED (4x)' if ENABLE_TTA else 'DISABLED'}")
    
    # Load test data and verify class mapping
    print(f"\nLoading test data from: {TEST_CSV_PATH}")
    test_df = pd.read_csv(TEST_CSV_PATH)
    print(f"Test set: {len(test_df)} images")
    
    transforms = get_transforms(IMG_SIZE)
    test_ds = HAM10000Dataset(test_df, IMG_DIR, transform=transforms['val'])
    
    # CRITICAL: Verify class mapping
    print("\nVerifying dataset label mapping:")
    dataset_label_map = test_ds.label_map
    for class_name, idx in sorted(dataset_label_map.items(), key=lambda x: x[1]):
        expected = CLASSES[idx] if idx < len(CLASSES) else "???"
        match = "OK" if class_name == expected else "MISMATCH"
        print(f"  [{match}] Index {idx}: {class_name}")
    
    # Dynamically find melanoma index instead of hardcoding
    MELANOMA_IDX = dataset_label_map.get('Melanoma')
    if MELANOMA_IDX is None:
        raise ValueError("'Melanoma' class not found in dataset label_map!")
    
    print(f"\nMelanoma class index: {MELANOMA_IDX}")
    print(f"Melanoma threshold: {MELANOMA_THRESHOLD}\n")
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True if use_amp else False
    )
    
    # Sequential Ensemble Inference
    print("="*70)
    print(" SEQUENTIAL ENSEMBLE INFERENCE")
    print("="*70)
    print("Loading one model at a time to conserve VRAM\n")
    
    num_samples = len(test_ds)
    ensemble_probs = np.zeros((num_samples, len(CLASSES)))
    y_true = None
    valid_models_count = 0
    
    for fold in range(5):
        path = os.path.join(CHECKPOINT_DIR, f"best_{MODEL_NAME}_fold{fold}.pth")
        
        if not os.path.exists(path):
            print(f"[Fold {fold}] WARNING: Checkpoint not found, skipping")
            continue
        
        print(f"\n[Fold {fold}] Loading model...")
        try:
            model = get_model(MODEL_NAME, num_classes=len(CLASSES), pretrained=False, use_film=USE_FILM)
            checkpoint = torch.load(path, map_location=device)
            sd = remap_swin_state_dict(checkpoint['model_state_dict'], model.state_dict())
            model.load_state_dict(sd)
            model = model.to(device)
            model.eval()
            
            fold_f1 = checkpoint.get('best_f1', 'N/A')
            print(f"[Fold {fold}] Loaded (CV F1: {fold_f1:.4f})" if isinstance(fold_f1, float) else f"[Fold {fold}] Loaded")
            
            valid_models_count += 1
            
            # Run inference for this fold
            fold_probs = []
            fold_labels = []
            
            with torch.no_grad():
                for images, metadata, labels in tqdm(test_loader, desc=f"[Fold {fold}] Inference", ncols=70, leave=False):
                    images = images.to(device)
                    metadata = metadata.to(device)
                    
                    if ENABLE_TTA:
                        probs = tta_inference(model, images, metadata, use_amp)
                    else:
                        if use_amp:
                            with autocast(device_type='cuda'):
                                outputs = model(images, metadata)
                        else:
                            outputs = model(images, metadata)
                        probs = F.softmax(outputs, dim=1).cpu().numpy()
                    
                    fold_probs.append(probs)
                    
                    # Only collect labels once
                    if valid_models_count == 1:
                        fold_labels.extend(labels.numpy())
            
            # Accumulate probabilities
            fold_probs = np.vstack(fold_probs)
            ensemble_probs += fold_probs
            
            if valid_models_count == 1:
                y_true = np.array(fold_labels)
            
            print(f"[Fold {fold}] Complete, clearing VRAM...")
            
            # Clear VRAM
            del model, checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"[Fold {fold}] ERROR: {e}")
            continue
    
    if valid_models_count == 0:
        raise RuntimeError("Failed to load any models!")
    
    print(f"\n{'='*70}")
    print(f"Successfully completed inference with {valid_models_count}/5 models")
    print(f"{'='*70}\n")
    
    # 1. Average probabilities across the ensemble
    y_probs = ensemble_probs / valid_models_count
    
    # 2. RAW PREDICTIONS (Standard 50% Argmax)
    y_preds_raw = np.argmax(y_probs, axis=1)
    
    # 3. THRESHOLD ADJUSTED PREDICTIONS (0.20 Melanoma Override)
    y_preds_adjusted = y_preds_raw.copy()
    melanoma_probs = y_probs[:, MELANOMA_IDX]
    old_mel_count = np.sum(y_preds_raw == MELANOMA_IDX)
    
    override_mask = (melanoma_probs >= MELANOMA_THRESHOLD) & (y_preds_raw != MELANOMA_IDX)
    y_preds_adjusted[override_mask] = MELANOMA_IDX
    
    new_mel_count = np.sum(y_preds_adjusted == MELANOMA_IDX)
    num_overrides = new_mel_count - old_mel_count
    
    override_stats = {
        'num_overrides': num_overrides,
        'old_count': old_mel_count,
        'new_count': new_mel_count
    }
    
    # 4. PER-CLASS OPTIMIZED WEIGHTS
    class_weights = None
    if CLASS_WEIGHTS_PATH and os.path.exists(CLASS_WEIGHTS_PATH):
        weights_data = np.load(CLASS_WEIGHTS_PATH, allow_pickle=True).item()
        class_weights = weights_data['weights']
        print(f"\nLoaded optimized class weights from {CLASS_WEIGHTS_PATH}")
        print(f"  Weights: {np.array2string(class_weights, precision=4, separator=', ')}")
        y_probs_scaled = y_probs * class_weights[np.newaxis, :]
        y_preds_optimized = np.argmax(y_probs_scaled, axis=1)
    elif CLASS_WEIGHTS_PATH:
        print(f"\nWARNING: Class weights file not found: {CLASS_WEIGHTS_PATH}")
        print("  Run 03_optimize_thresholds.py first. Skipping optimized evaluation.")

    # ==========================================
    #               SAVE RESULTS
    # ==========================================
    print("Generating Reports & Plots...")
    
    # Process RAW
    raw_dir = os.path.join(OUTPUT_DIR, "raw")
    evaluate_and_save(
        y_true, y_probs, y_preds_raw, 
        out_dir=raw_dir, 
        title_prefix="Raw Baseline (Standard Argmax)"
    )
    
    # Process ADJUSTED
    adj_dir = os.path.join(OUTPUT_DIR, f"threshold_{MELANOMA_THRESHOLD}")
    evaluate_and_save(
        y_true, y_probs, y_preds_adjusted, 
        out_dir=adj_dir, 
        title_prefix=f"Threshold Adjusted (Melanoma >= {MELANOMA_THRESHOLD})",
        is_adjusted=True,
        override_stats=override_stats
    )
    
    # Process OPTIMIZED WEIGHTS
    if class_weights is not None:
        opt_dir = os.path.join(OUTPUT_DIR, "optimized_weights")
        evaluate_and_save(
            y_true, y_probs, y_preds_optimized, 
            out_dir=opt_dir, 
            title_prefix="Per-Class Optimized Weights"
        )
    
    print(f"\n{'='*70}")
    print(" EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nCheck {OUTPUT_DIR}/ for both raw and adjusted result folders!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()