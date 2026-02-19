import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, 
    classification_report, accuracy_score, f1_score
)
from torch.utils.data import DataLoader
from torch.amp import autocast  # Better import
from itertools import cycle
import os
from tqdm import tqdm

from dataset_factory import HAM10000Dataset, get_transforms
from model_factory import get_model

# ==========================================
#              CONFIGURATION
# ==========================================
MODEL_NAME = "swinv2_large_window12to24_192to384_22kft1k"
IMG_SIZE = 384
BATCH_SIZE = 8   
NUM_WORKERS = 0    # Windows-safe

TEST_CSV_PATH = "./data/test_split.csv" 
IMG_DIR = "./data/all_images"
CHECKPOINT_DIR = "./checkpoints_swin_large"


CLASSES = [
    'Melanomaanocytic Nevi (moles)',                      # nv
    'Melanoma',                                           # mel
    'Benign Keratosis-like Lesions',                      # bkl
    'Basal Cell Carcinoma',                               # bcc
    'Actinic Keratoses & Intraepithelial Carcinoma',      # akiec
    'Vascular Lesions',                                   # vasc
    'Dermatofibroma'                                      # df
]

def load_ensemble(device):
    """Load all 5 fold models for ensemble prediction"""
    print(f"\nLoading ensemble models from: {CHECKPOINT_DIR}")
    print("-" * 50)
    
    if not os.path.exists(CHECKPOINT_DIR):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {CHECKPOINT_DIR}\n"
            f"Please make sure you've copied the checkpoint folder to your laptop."
        )
    
    models = []
    fold_f1s = []
    
    for fold in range(5):
        path = os.path.join(CHECKPOINT_DIR, f"best_{MODEL_NAME}_fold{fold}.pth")
        
        if not os.path.exists(path):
            print(f"    WARNING: Fold {fold} checkpoint not found!")
            print(f"      Expected at: {path}")
            continue
        
        try:
            # Load model
            model = get_model(MODEL_NAME, num_classes=len(CLASSES))
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            
            models.append(model)
            fold_f1 = checkpoint.get('best_f1', 'N/A')
            fold_f1s.append(fold_f1)
            
            print(f"   Fold {fold}: Loaded (CV F1: {fold_f1:.4f})" if isinstance(fold_f1, float) else f"   Fold {fold}: Loaded")
            
        except Exception as e:
            print(f"   ERROR loading fold {fold}: {e}")
            continue
    
    print("-" * 50)
    
    if not models:
        raise RuntimeError(
            f"Failed to load any models!\n"
            f"Check that:\n"
            f"  1. Checkpoint files exist in {CHECKPOINT_DIR}\n"
            f"  2. MODEL_NAME '{MODEL_NAME}' matches your checkpoint filenames\n"
            f"  3. Files aren't corrupted"
        )
    
    print(f" Successfully loaded {len(models)}/5 models for ensemble\n")
    
    if len(models) < 5:
        print(f"  Note: Using {len(models)}-model ensemble (some folds missing)")
    
    return models

def plot_confusion_matrix(y_true, y_pred, classes, save_path='test_confusion_matrix.png'):
    """Plot and save normalized confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Shorten labels for better display
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
    plt.title('Normalized Confusion Matrix\n(Ensemble Test Set)', fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved {save_path}")

def plot_multiclass_roc(y_true, y_probs, classes, save_path='test_roc_curve.png'):
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
    plt.title('Multi-Class ROC Curves\n(Ensemble Test Set)', fontsize=14, pad=20)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved {save_path}")

def main():
    # Device detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f" Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        use_amp = True
    else:
        device = torch.device("cpu")
        print("  No GPU detected, using CPU (will be slow)")
        use_amp = False
    
    print("\n" + "="*60)
    print(" HAM10000 Test Set Evaluation - Ensemble Inference")
    print("="*60)
    print(f"Model:       {MODEL_NAME}")
    print(f"Device:      {device}")
    print(f"Batch Size:  {BATCH_SIZE}")
    print(f"Mixed Precision: {use_amp}")
    print("="*60)
    
    # 1. Load test data
    if not os.path.exists(TEST_CSV_PATH):
        raise FileNotFoundError(f"Test CSV not found: {TEST_CSV_PATH}")
    
    if not os.path.exists(IMG_DIR):
        raise FileNotFoundError(f"Image directory not found: {IMG_DIR}")
    
    print(f"\nLoading test data from: {TEST_CSV_PATH}")
    test_df = pd.read_csv(TEST_CSV_PATH)
    print(f" Test set size: {len(test_df)} images")
    
    print("\nClass distribution in test set:")
    for cls, count in test_df['dx'].value_counts().items():
        print(f"  {cls}: {count}")
    
    # Create dataset
    transforms = get_transforms(IMG_SIZE)
    test_ds = HAM10000Dataset(test_df, IMG_DIR, transform=transforms['val'])
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True if use_amp else False
    )
    
    # 2. Load ensemble models
    models = load_ensemble(device)
    
    # 3. Run inference
    all_probs = []
    all_labels = []
    
    print("Running ensemble inference on test set...")
    print(f"Processing {len(test_loader)} batches...\n")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", ncols=80):
            images = images.to(device)
            labels = labels.cpu().numpy()
            
            # Initialize probability accumulator
            batch_probs = torch.zeros((images.size(0), len(CLASSES))).to(device)
            
            # Ensemble prediction with conditional AMP
            if use_amp:
                with autocast(device_type='cuda'):
                    for model in models:
                        outputs = model(images)
                        batch_probs += F.softmax(outputs, dim=1)
            else:
                for model in models:
                    outputs = model(images)
                    batch_probs += F.softmax(outputs, dim=1)
            
            # Average probabilities across models
            batch_probs /= len(models)
            
            all_probs.extend(batch_probs.cpu().numpy())
            all_labels.extend(labels)
    
    # 4. Process results
    y_probs = np.array(all_probs)
    y_true = np.array(all_labels)
    y_preds = np.argmax(y_probs, axis=1)
    
    # Get class names in correct order
    idx_to_class = {v: k for k, v in test_ds.label_map.items()}
    class_names = [idx_to_class[i] for i in range(len(CLASSES))]
    
    # 5. Calculate and display metrics
    print("\n" + "="*60)
    print(" ENSEMBLE TEST SET RESULTS")
    print("="*60)
    
    test_acc = accuracy_score(y_true, y_preds)
    test_f1_macro = f1_score(y_true, y_preds, average='macro')
    test_f1_weighted = f1_score(y_true, y_preds, average='weighted')
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:           {test_acc:.4f}")
    print(f"  F1 Score (Macro):   {test_f1_macro:.4f}")
    print(f"  F1 Score (Weighted): {test_f1_weighted:.4f}")
    
    print("\n" + "="*60)
    print(" PER-CLASS CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_preds, target_names=class_names, digits=4))
    
    # 6. Generate and save plots
    print("\n" + "="*60)
    print(" Generating visualizations...")
    print("="*60)
    
    plot_confusion_matrix(y_true, y_preds, class_names)
    plot_multiclass_roc(y_true, y_probs, class_names)
    
    # 7. Save detailed results
    results = {
        'predictions': y_preds,
        'true_labels': y_true,
        'probabilities': y_probs,
        'class_names': class_names,
        'metrics': {
            'accuracy': float(test_acc),
            'f1_macro': float(test_f1_macro),
            'f1_weighted': float(test_f1_weighted)
        },
        'model_name': MODEL_NAME,
        'num_models': len(models)
    }
    
    np.save('test_results.npy', results)
    print(f"   Saved test_results.npy")
    
    print("\n" + "="*60)
    print("  EVALUATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  • test_confusion_matrix.png")
    print("  • test_roc_curve.png")
    print("  • test_results.npy")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user")
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()