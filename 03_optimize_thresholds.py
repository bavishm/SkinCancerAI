"""
03_optimize_thresholds.py
=========================
Per-class threshold (weight) optimization using validation predictions.

Loads the 5-fold out-of-fold predictions from training, stacks them,
and uses differential evolution to find per-class scaling weights that
maximize macro F1 on the combined validation set.

The optimized weights are saved as a .npy file that can be loaded by
the evaluation scripts at test time.

Usage:
    python 03_optimize_thresholds.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import f1_score, accuracy_score, classification_report
import os
import time

# ==========================================
#               CONFIGURATION
# ==========================================
CHECKPOINT_DIR = "./checkpoints_swinv2_focal_film_aug"  # Where predictions_fold*.csv are
N_FOLDS = 5
OUTPUT_PATH = os.path.join(CHECKPOINT_DIR, "class_weights_optimized.npy")  # Saved alongside CSVs

CLASSES = [
    'Melanomaanocytic Nevi (moles)',                      # 0
    'Melanoma',                                           # 1
    'Benign Keratosis-like Lesions',                      # 2
    'Basal Cell Carcinoma',                               # 3
    'Actinic Keratoses & Intraepithelial Carcinoma',      # 4
    'Vascular Lesions',                                   # 5
    'Dermatofibroma'                                      # 6
]
SHORT_NAMES = ['Nevi', 'Melanoma', 'BKL', 'BCC', 'AKIEC', 'Vascular', 'Derm.']
NUM_CLASSES = len(CLASSES)

# Search bounds for each class weight
WEIGHT_BOUNDS = [(0.3, 3.0)] * NUM_CLASSES

# Differential evolution parameters
DE_MAXITER = 200       # Maximum iterations
DE_POPSIZE = 30        # Population size multiplier
DE_TOL = 1e-7          # Convergence tolerance
DE_SEED = 42           # Reproducibility
N_RESTARTS = 5         # Number of independent optimization runs


def load_validation_predictions():
    """Load and stack all 5-fold out-of-fold predictions."""
    all_probs = []
    all_labels = []
    
    for fold in range(N_FOLDS):
        csv_path = os.path.join(CHECKPOINT_DIR, f"predictions_fold{fold}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Extract probability columns
        prob_cols = [f"prob_{i}" for i in range(NUM_CLASSES)]
        probs = df[prob_cols].values
        labels = df['true_label'].values
        
        all_probs.append(probs)
        all_labels.append(labels)
        
        print(f"  Fold {fold}: {len(df)} samples loaded")
    
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    
    return all_probs, all_labels


def objective_macro_f1(weights, probs, true_labels):
    """Objective function: negative macro F1 (minimize)."""
    weights = np.array(weights)
    scaled = probs * weights[np.newaxis, :]  # [N, 7] * [1, 7]
    preds = np.argmax(scaled, axis=1)
    return -f1_score(true_labels, preds, average='macro')


def objective_combined(weights, probs, true_labels):
    """Combined objective: -(0.7 * macro_F1 + 0.3 * accuracy).
    Prevents macro F1 optimization from tanking overall accuracy."""
    weights = np.array(weights)
    scaled = probs * weights[np.newaxis, :]
    preds = np.argmax(scaled, axis=1)
    macro_f1 = f1_score(true_labels, preds, average='macro')
    acc = accuracy_score(true_labels, preds)
    return -(0.7 * macro_f1 + 0.3 * acc)


def evaluate_weights(weights, probs, true_labels, label=""):
    """Evaluate a set of weights and print detailed results."""
    weights = np.array(weights)
    scaled = probs * weights[np.newaxis, :]
    preds = np.argmax(scaled, axis=1)
    
    acc = accuracy_score(true_labels, preds)
    f1_macro = f1_score(true_labels, preds, average='macro')
    f1_weighted = f1_score(true_labels, preds, average='weighted')
    f1_per_class = f1_score(true_labels, preds, average=None)
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Accuracy:           {acc:.4f}")
    print(f"  F1 Score (Macro):   {f1_macro:.4f}")
    print(f"  F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"\n  Per-class F1 scores:")
    print(f"  {'Class':15s} {'Weight':>8s} {'F1':>8s}")
    print(f"  {'-'*33}")
    for i in range(NUM_CLASSES):
        print(f"  {SHORT_NAMES[i]:15s} {weights[i]:8.4f} {f1_per_class[i]:8.4f}")
    
    print(f"\n  Full report:")
    print(classification_report(true_labels, preds, target_names=SHORT_NAMES, digits=4))
    
    return acc, f1_macro, f1_weighted


def main():
    start_time = time.time()
    
    print("=" * 60)
    print("  PER-CLASS WEIGHT OPTIMIZATION")
    print("=" * 60)
    print(f"\nLoading validation predictions from: {CHECKPOINT_DIR}/")
    
    probs, labels = load_validation_predictions()
    print(f"\n  Total validation samples: {len(labels)}")
    print(f"  Class distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    [{int(u)}] {SHORT_NAMES[int(u)]:15s}: {c}")
    
    # ==========================================
    #  BASELINE: Uniform weights (standard argmax)
    # ==========================================
    uniform_weights = np.ones(NUM_CLASSES)
    print("\n" + "=" * 60)
    evaluate_weights(uniform_weights, probs, labels, "BASELINE (Uniform Weights = 1.0)")
    
    # ==========================================
    #  OPTIMIZE: Macro F1 objective
    # ==========================================
    print("\n" + "=" * 60)
    print("  RUNNING OPTIMIZATION (Differential Evolution)")
    print("=" * 60)
    print(f"  Objective: Maximize 0.7*macro_F1 + 0.3*accuracy")
    print(f"  Search bounds per class: {WEIGHT_BOUNDS[0]}")
    print(f"  Max iterations: {DE_MAXITER}")
    print(f"  Population size: {DE_POPSIZE}x{NUM_CLASSES} = {DE_POPSIZE * NUM_CLASSES}")
    print(f"  Restarts: {N_RESTARTS}")
    
    best_result = None
    best_score = float('inf')
    
    for restart in range(N_RESTARTS):
        seed = DE_SEED + restart * 1000
        print(f"\n  --- Restart {restart + 1}/{N_RESTARTS} (seed={seed}) ---")
        
        result = differential_evolution(
            objective_combined,
            bounds=WEIGHT_BOUNDS,
            args=(probs, labels),
            maxiter=DE_MAXITER,
            popsize=DE_POPSIZE,
            tol=DE_TOL,
            seed=seed,
            workers=-1,       # Use all CPU cores
            updating='deferred',  # Required for parallel workers
            disp=False,
            polish=True       # L-BFGS-B polish at the end
        )
        
        print(f"  Score: {-result.fun:.6f} | Converged: {result.success} | Iterations: {result.nit}")
        
        if result.fun < best_score:
            best_score = result.fun
            best_result = result
    
    optimal_weights = best_result.x
    
    # Normalize: set max weight to 1.0 so we're only boosting minority classes
    # (optional, doesn't change argmax outcome)
    # optimal_weights = optimal_weights / optimal_weights.max()
    
    # ==========================================
    #  EVALUATE OPTIMIZED WEIGHTS
    # ==========================================
    acc, f1_macro, f1_weighted = evaluate_weights(
        optimal_weights, probs, labels, 
        "OPTIMIZED WEIGHTS (on validation set)"
    )
    
    # ==========================================
    #  COMPARE BASELINE vs OPTIMIZED
    # ==========================================
    baseline_preds = np.argmax(probs, axis=1)
    baseline_f1 = f1_score(labels, baseline_preds, average='macro')
    baseline_acc = accuracy_score(labels, baseline_preds)
    baseline_f1w = f1_score(labels, baseline_preds, average='weighted')
    
    print(f"\n{'='*60}")
    print(f"  SUMMARY: BASELINE vs OPTIMIZED")
    print(f"{'='*60}")
    print(f"  {'Metric':20s} {'Baseline':>10s} {'Optimized':>10s} {'Delta':>10s}")
    print(f"  {'-'*52}")
    print(f"  {'Accuracy':20s} {baseline_acc:10.4f} {acc:10.4f} {acc - baseline_acc:+10.4f}")
    print(f"  {'F1 Macro':20s} {baseline_f1:10.4f} {f1_macro:10.4f} {f1_macro - baseline_f1:+10.4f}")
    print(f"  {'F1 Weighted':20s} {baseline_f1w:10.4f} {f1_weighted:10.4f} {f1_weighted - baseline_f1w:+10.4f}")
    
    # ==========================================
    #  SAVE WEIGHTS
    # ==========================================
    save_data = {
        'weights': optimal_weights,
        'class_names': CLASSES,
        'short_names': SHORT_NAMES,
        'validation_metrics': {
            'accuracy': float(acc),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted)
        },
        'baseline_metrics': {
            'accuracy': float(baseline_acc),
            'f1_macro': float(baseline_f1),
            'f1_weighted': float(baseline_f1w)
        },
        'source_checkpoint_dir': CHECKPOINT_DIR,
        'optimization_params': {
            'bounds': WEIGHT_BOUNDS,
            'maxiter': DE_MAXITER,
            'popsize': DE_POPSIZE,
            'n_restarts': N_RESTARTS,
            'objective': '0.7*macro_f1 + 0.3*accuracy'
        }
    }
    
    np.save(OUTPUT_PATH, save_data)
    print(f"\n  Optimized weights saved to: {OUTPUT_PATH}")
    print(f"  Weights: {np.array2string(optimal_weights, precision=4, separator=', ')}")
    
    elapsed = time.time() - start_time
    print(f"\n  Time elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
