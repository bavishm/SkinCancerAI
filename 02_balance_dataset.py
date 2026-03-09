"""
02_balance_dataset.py
=====================
Offline class balancing via augmentation.

For each fold's training split:
1. Finds the majority class count
2. For every minority class, generates augmented copies until it matches
3. Saves augmented images to data/augmented_fold{N}/
4. Produces data/balanced_fold{N}.csv with original + augmented entries

Run this ONCE before training. Training then loads the balanced CSVs
with shuffle=True (no WeightedRandomSampler needed).
"""

import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count
from functools import partial

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

# Heavy augmentation pipeline for generating synthetic training images
# These are applied ONCE per generated image and saved to disk
augmentation_pipeline = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    
    # Geometric
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.7),
    A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=0, p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
    
    # Color/Texture
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=15, p=0.5),
    A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7)),
        A.GaussNoise(),
    ], p=0.3),
])


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
        task: tuple of (src_img_id, aug_img_id, aug_path, metadata_dict)
    Returns:
        metadata_dict (the new row for the balanced CSV)
    """
    src_img_id, aug_img_id, aug_path, metadata = task
    
    # Load source image
    image = load_image(src_img_id)
    
    # Apply augmentation (each worker has its own random state)
    augmented = augmentation_pipeline(image=image)
    aug_image = augmented['image']
    
    # Save as BGR for OpenCV compatibility
    cv2.imwrite(aug_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return metadata


def balance_fold(fold, full_df):
    """Generate augmented images for one fold's training split."""
    
    train_df = full_df[full_df['fold'] != fold].reset_index(drop=True)
    
    # Per-class counts
    class_counts = train_df['dx'].value_counts()
    max_count = class_counts.max()
    majority_class = class_counts.idxmax()
    
    print(f"\n{'='*60}")
    print(f"FOLD {fold} — Balancing to {max_count} per class (majority: {majority_class[:20]}...)")
    print(f"{'='*60}")
    print(f"Original training set: {len(train_df)} images")
    print(f"\nClass distribution BEFORE balancing:")
    for cls_name, count in class_counts.items():
        cls_idx = LABEL_MAP[cls_name]
        deficit = max_count - count
        print(f"  [{cls_idx}] {SHORT_NAMES[cls_idx]:15s}: {count:5d}  (need +{deficit})")
    
    # Create fold-specific augmentation directory
    fold_aug_dir = os.path.join(AUG_DIR, f"fold{fold}")
    os.makedirs(fold_aug_dir, exist_ok=True)
    
    # Build ALL tasks upfront for parallel execution
    tasks = []
    
    for cls_name, count in class_counts.items():
        deficit = max_count - count
        if deficit == 0:
            continue  # Majority class, skip
        
        cls_idx = LABEL_MAP[cls_name]
        cls_df = train_df[train_df['dx'] == cls_name].reset_index(drop=True)
        short = SHORT_NAMES[cls_idx]
        
        # Pre-select source images randomly
        src_indices = np.random.randint(0, len(cls_df), size=deficit)
        
        for i, src_idx in enumerate(src_indices):
            src_row = cls_df.iloc[src_idx]
            src_img_id = str(src_row['image_id'])
            aug_img_id = f"aug_fold{fold}_{short}_{i:05d}"
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
            tasks.append((src_img_id, aug_img_id, aug_path, metadata))
    
    total_generated = len(tasks)
    print(f"\n  Generating {total_generated} augmented images using {NUM_WORKERS} workers...")
    
    # Parallel augmentation
    new_rows = []
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
    for cls_name, count in balanced_counts.items():
        cls_idx = LABEL_MAP[cls_name]
        print(f"  [{cls_idx}] {SHORT_NAMES[cls_idx]:15s}: {count:5d}")
    
    # Save balanced CSV
    csv_path = os.path.join(DATA_DIR, f"balanced_fold{fold}.csv")
    balanced_df.to_csv(csv_path, index=False)
    
    print(f"\n  Total: {len(balanced_df)} images ({len(train_df)} original + {total_generated} augmented)")
    print(f"  Saved: {csv_path}")
    print(f"  Augmented images saved to: {fold_aug_dir}/")
    
    # Build per-class stats dict for summary CSV
    fold_stats = {'fold': fold, 'original_total': len(train_df), 'augmented_total': total_generated, 'balanced_total': len(balanced_df)}
    for cls_name, count in class_counts.items():
        cls_idx = LABEL_MAP[cls_name]
        short = SHORT_NAMES[cls_idx]
        deficit = max_count - count
        fold_stats[f'{short}_original'] = int(count)
        fold_stats[f'{short}_augmented'] = int(deficit)
        fold_stats[f'{short}_balanced'] = int(max_count)
    
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
    
    os.makedirs(AUG_DIR, exist_ok=True)
    
    total_images = 0
    total_augmented = 0
    all_fold_stats = []
    
    for fold in range(N_FOLDS):
        fold_total, fold_aug, fold_stats = balance_fold(fold, full_df)
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
