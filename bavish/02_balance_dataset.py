"""
02_balance_dataset.py
=====================
Offline class augmentation via per-class multipliers.

For each fold's training split:
1. Uses CLASS_AUG_MULTIPLIERS to set target count per class
2. Generates augmented copies until target count is reached
3. Saves augmented images to data/augmented/fold{N}/
4. Produces data/balanced_fold{N}.csv with original + augmented entries

Rerun behavior:
- If DELETE_OLD_FOLD_AUG_FILES=True, old generated files in each fold directory
    are removed before new augmentation starts.

Run this ONCE before training. Training then loads the balanced CSVs
with shuffle=True (no WeightedRandomSampler needed).
"""

import os
import shutil
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count

# ==========================================
#               CONFIGURATION
# ==========================================
DATA_DIR = "./data"
IMG_DIR = "./data/all_images"
AUG_DIR = "./data/augmented"  # Root for augmented images
FOLDS_CSV = os.path.join(DATA_DIR, "train_folds.csv")
IMG_SIZE = 384
N_FOLDS = 5
NUM_WORKERS = min(36, cpu_count())  # leave a few cores free
SEED = 42

# Professor-style per-class augmentation multipliers.
# Interpretation: target_count = ceil(multiplier * original_count).
# Provide in class-index order [0..6].
# If the list is shorter than 7, missing classes default to multiplier=1.0.
# Example style: [2, 2, 3, 5, 6]
CLASS_AUG_MULTIPLIERS = [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]

# On rerun, delete old generated files in each fold directory before regenerating.
DELETE_OLD_FOLD_AUG_FILES = True

# Label map (must match dataset_factory.py)
LABEL_MAP = {
    'Melanomaanocytic Nevi (moles)': 0,
    'Melanoma': 1,
    'Benign Keratosis-like Lesions': 2,
    'Basal Cell Carcinoma': 3,
    'Actinic Keratoses & Intraepithelial Carcinoma': 4,
    'Vascular Lesions': 5,
    'Dermatofibroma': 6
}
SHORT_NAMES = {0: 'Nevi', 1: 'Melanoma', 2: 'BKL', 3: 'BCC', 4: 'AKIEC', 5: 'Vascular', 6: 'Dermatofibroma'}

# Augmentation pipeline for generating synthetic training images.
# Applied once per generated image and saved to disk.
augmentation_pipeline = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    
    # Geometric
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.35),
    A.Rotate(limit=25, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.12, rotate_limit=0, p=0.35),
    
    # Color/Texture
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5)),
        A.GaussNoise(),
    ], p=0.2),
])


def get_multiplier_for_class(class_idx):
    """Return multiplier for class index, defaulting to 1.0 if not provided."""
    if class_idx < len(CLASS_AUG_MULTIPLIERS):
        return float(CLASS_AUG_MULTIPLIERS[class_idx])
    return 1.0


def load_image(img_id):
    """Load an image by ID from the image directory."""
    if not img_id.endswith('.jpg'):
        img_id = img_id + '.jpg'
    path = os.path.join(IMG_DIR, img_id)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _augment_single(task):
    """Worker function for multiprocessing. Loads, augments, and saves one image.
    
    Args:
        task: tuple of (src_img_id, aug_path, metadata_dict)
    Returns:
        metadata_dict (the new row for the balanced CSV)
    """
    src_img_id, aug_path, metadata = task
    
    # Load source image
    image = load_image(src_img_id)
    
    # Apply augmentation (each worker has its own random state)
    augmented = augmentation_pipeline(image=image)
    aug_image = augmented['image']
    
    # Save as BGR for OpenCV compatibility
    cv2.imwrite(aug_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return metadata


def balance_fold(fold, full_df, rng):
    """Generate augmented images for one fold's training split."""
    
    train_df = full_df[full_df['fold'] != fold].reset_index(drop=True)
    
    # Per-class counts
    class_counts = train_df['dx'].value_counts()
    
    print(f"\n{'='*60}")
    print(f"FOLD {fold} — Per-class multiplier augmentation")
    print(f"{'='*60}")
    print(f"Original training set: {len(train_df)} images")
    print(f"\nClass distribution BEFORE balancing:")
    for cls_name, cls_idx in sorted(LABEL_MAP.items(), key=lambda kv: kv[1]):
        count = int(class_counts.get(cls_name, 0))
        mult = get_multiplier_for_class(cls_idx)
        target = max(count, int(np.ceil(count * mult)))
        deficit = target - count
        print(
            f"  [{cls_idx}] {SHORT_NAMES[cls_idx]:15s}: {count:5d}  "
            f"(mult={mult:.2f}, target={target}, need +{deficit})"
        )
    
    # Create fold-specific augmentation directory
    fold_aug_dir = os.path.join(AUG_DIR, f"fold{fold}")

    if DELETE_OLD_FOLD_AUG_FILES and os.path.exists(fold_aug_dir):
        shutil.rmtree(fold_aug_dir)

    os.makedirs(fold_aug_dir, exist_ok=True)
    
    # Build ALL tasks upfront for parallel execution
    tasks = []
    
    for cls_name, cls_idx in sorted(LABEL_MAP.items(), key=lambda kv: kv[1]):
        count = int(class_counts.get(cls_name, 0))
        multiplier = get_multiplier_for_class(cls_idx)
        target_count = max(count, int(np.ceil(count * multiplier)))
        deficit = target_count - count
        if deficit <= 0:
            continue

        if count == 0:
            print(f"  [WARNING] Class {cls_name} has 0 samples in this fold; skipping.")
            continue

        cls_df = train_df[train_df['dx'] == cls_name].reset_index(drop=True)
        short = SHORT_NAMES[cls_idx]
        
        # Pre-select source images randomly
        src_indices = rng.integers(0, len(cls_df), size=deficit)
        
        for i, src_idx in enumerate(src_indices):
            src_row = cls_df.iloc[int(src_idx)]
            src_img_id = str(src_row['image_id'])
            aug_img_id = f"aug_fold{fold}_cls{cls_idx}_{short}_{i:05d}"
            aug_path = os.path.join(fold_aug_dir, f"{aug_img_id}.jpg")
            
            metadata = {
                'lesion_id': src_row['lesion_id'],
                'image_id': aug_img_id,
                'dx': cls_name,
                'dx_type': src_row['dx_type'],
                'age': src_row['age'],
                'sex': src_row['sex'],
                'localization': src_row['localization'],
                'fold': -1,
                'is_augmented': True,
                'source_image': src_img_id
            }
            tasks.append((src_img_id, aug_path, metadata))
    
    total_generated = len(tasks)
    print(f"\n  Generating {total_generated} augmented images using {NUM_WORKERS} workers...")
    
    # Parallel augmentation
    new_rows = []
    if total_generated > 0:
        with Pool(processes=NUM_WORKERS) as pool:
            for result in tqdm(pool.imap_unordered(_augment_single, tasks, chunksize=64),
                               total=total_generated, desc=f"  Fold {fold}", ncols=100):
                new_rows.append(result)
    
    # Build balanced CSV: original training images + augmented images
    aug_df = pd.DataFrame(new_rows)
    
    # Add 'is_augmented' and 'source_image' columns to original df
    train_df = train_df.copy()
    train_df['is_augmented'] = False
    train_df['source_image'] = ''
    
    balanced_df = pd.concat([train_df, aug_df], ignore_index=True)
    
    # Verify balance
    print(f"\n  Class distribution AFTER balancing:")
    balanced_counts = balanced_df['dx'].value_counts()
    for cls_name, cls_idx in sorted(LABEL_MAP.items(), key=lambda kv: kv[1]):
        count = int(balanced_counts.get(cls_name, 0))
        print(f"  [{cls_idx}] {SHORT_NAMES[cls_idx]:15s}: {count:5d}")
    
    # Save balanced CSV
    csv_path = os.path.join(DATA_DIR, f"balanced_fold{fold}.csv")
    balanced_df.to_csv(csv_path, index=False)
    
    print(f"\n  Total: {len(balanced_df)} images ({len(train_df)} original + {total_generated} augmented)")
    print(f"  Saved: {csv_path}")
    print(f"  Augmented images saved to: {fold_aug_dir}/")
    
    # Build per-class stats dict for summary CSV
    multipliers_serialized = "[" + ", ".join(
        [str(get_multiplier_for_class(i)) for i in range(len(LABEL_MAP))]
    ) + "]"
    fold_stats = {
        'fold': fold,
        'class_aug_multipliers': multipliers_serialized,
        'original_total': len(train_df),
        'augmented_total': total_generated,
        'balanced_total': len(balanced_df)
    }
    for cls_name, cls_idx in sorted(LABEL_MAP.items(), key=lambda kv: kv[1]):
        count = int(class_counts.get(cls_name, 0))
        short = SHORT_NAMES[cls_idx]
        multiplier = get_multiplier_for_class(cls_idx)
        target_count = max(count, int(np.ceil(count * multiplier)))
        deficit = max(0, target_count - count)
        fold_stats[f'{short}_multiplier'] = float(multiplier)
        fold_stats[f'{short}_original'] = int(count)
        fold_stats[f'{short}_target'] = int(target_count)
        fold_stats[f'{short}_augmented'] = int(deficit)
        fold_stats[f'{short}_balanced'] = int(balanced_counts.get(cls_name, 0))
    
    return len(balanced_df), total_generated, fold_stats


def main():
    start_time = time.time()
    
    print("=" * 60)
    print("  OFFLINE CLASS BALANCING VIA AUGMENTATION")
    print("=" * 60)
    
    if not os.path.exists(FOLDS_CSV):
        raise FileNotFoundError(f"Run 01_split_data.py first! Missing: {FOLDS_CSV}")
    
    full_df = pd.read_csv(FOLDS_CSV)
    print(f"Loaded {len(full_df)} images from {FOLDS_CSV}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Output directory: {AUG_DIR}/")
    print(f"Delete old fold files on rerun: {DELETE_OLD_FOLD_AUG_FILES}")
    print(f"Class multipliers (by class index): {CLASS_AUG_MULTIPLIERS}")
    
    os.makedirs(AUG_DIR, exist_ok=True)
    
    total_images = 0
    total_augmented = 0
    all_fold_stats = []
    
    for fold in range(N_FOLDS):
        rng = np.random.default_rng(SEED + fold)
        fold_total, fold_aug, fold_stats = balance_fold(fold, full_df, rng)
        total_images += fold_total
        total_augmented += fold_aug
        all_fold_stats.append(fold_stats)
    
    # Save augmentation summary CSV
    summary_df = pd.DataFrame(all_fold_stats)
    summary_path = os.path.join(DATA_DIR, "augmentation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Augmentation summary saved to: {summary_path}")
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"  BALANCING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total balanced images across all folds: {total_images}")
    print(f"  Total augmented images generated: {total_augmented}")
    print(f"  Time elapsed: {elapsed/60:.1f} minutes")
    print(f"\n  Generated files:")
    for fold in range(N_FOLDS):
        csv_path = os.path.join(DATA_DIR, f"balanced_fold{fold}.csv")
        aug_dir = os.path.join(AUG_DIR, f"fold{fold}")
        n_aug = len(os.listdir(aug_dir)) if os.path.exists(aug_dir) else 0
        print(f"    Fold {fold}: {csv_path} ({n_aug} augmented images)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
