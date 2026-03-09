import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================
#         METADATA ENCODING CONSTANTS
# ==========================================
# Fixed ordering for reproducible one-hot encoding across train/val/test
LOCALIZATION_CLASSES = [
    'abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot',
    'genital', 'hand', 'lower extremity', 'neck', 'scalp',
    'trunk', 'unknown', 'upper extremity'
]
METADATA_DIM = 1 + 1 + len(LOCALIZATION_CLASSES)  # age + sex + localization one-hot = 17

def encode_metadata(age, sex, localization):
    """Encode patient metadata into a fixed-size float vector.
    
    Returns: np.array of shape (METADATA_DIM,) = (17,)
        [0]    : age (normalized 0-1, NaN → 0.5)
        [1]    : sex (male=0, female=1, unknown=0.5)
        [2-16] : localization one-hot (15 categories)
    """
    meta = np.zeros(METADATA_DIM, dtype=np.float32)
    
    # Age: normalize to [0, 1], NaN → 0.5 (population median is ~50)
    if pd.isna(age) or age is None:
        meta[0] = 0.5
    else:
        meta[0] = float(age) / 85.0  # max age in dataset is 85
    
    # Sex: binary + unknown
    if sex == 'female':
        meta[1] = 1.0
    elif sex == 'male':
        meta[1] = 0.0
    else:
        meta[1] = 0.5  # unknown
    
    # Localization: one-hot
    loc_str = str(localization).lower().strip() if pd.notna(localization) else 'unknown'
    if loc_str in LOCALIZATION_CLASSES:
        loc_idx = LOCALIZATION_CLASSES.index(loc_str)
    else:
        loc_idx = LOCALIZATION_CLASSES.index('unknown')
    meta[2 + loc_idx] = 1.0
    
    return meta

# Need pandas for NaN checks in encode_metadata
import pandas as pd

class HAM10000Dataset(Dataset):
    def __init__(self, df, img_dir, transform=None, aug_img_dir=None):
        """
        Args:
            df: DataFrame with image_id, dx, age, sex, localization columns
            img_dir: Path to original images (data/all_images/)
            transform: Albumentations transform pipeline
            aug_img_dir: Path to augmented images for this fold (data/augmented/fold{N}/).
                         If None, all images are loaded from img_dir.
        """
        self.df = df
        self.img_dir = img_dir
        self.aug_img_dir = aug_img_dir
        self.transform = transform
        
        # Mapping specific long-form CSV labels to integers 0-6.
        # This matches the exact strings found in the HAM10000 metadata.
        self.label_map = {
            'Melanomaanocytic Nevi (moles)': 0,             # nv
            'Melanoma': 1,                                  # mel
            'Benign Keratosis-like Lesions': 2,             # bkl
            'Basal Cell Carcinoma': 3,                      # bcc
            'Actinic Keratoses & Intraepithelial Carcinoma': 4, # akiec
            'Vascular Lesions': 5,                          # vasc
            'Dermatofibroma': 6                             # df
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        img_id = str(row['image_id'])

        # Append .jpg if missing
        if not img_id.endswith('.jpg'):
            img_id = img_id + '.jpg'
        
        # Route to correct directory: augmented images go to aug_img_dir, originals to img_dir
        is_aug = bool(row.get('is_augmented', False))
        if is_aug and self.aug_img_dir:
            img_path = os.path.join(self.aug_img_dir, img_id)
        else:
            img_path = os.path.join(self.img_dir, img_id)
        
        # Load Image using OpenCV (High performance for servers)
        image = cv2.imread(img_path)
        
        # Error handling: If image is corrupt or missing
        if image is None:
            raise FileNotFoundError(f"CRITICAL: Image not found at {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Augmentations (Albumentations)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Map the string label to the integer
        label_str = row['dx']
        
        if label_str not in self.label_map:
            raise KeyError(f"Label '{label_str}' not found in label_map.")
            
        label = self.label_map[label_str]
        
        # Encode metadata (age, sex, localization) for FiLM conditioning
        metadata = encode_metadata(row.get('age'), row.get('sex'), row.get('localization'))
        
        return image, torch.tensor(metadata, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def get_transforms(img_size=384):
    
    # Returns a dictionary of transformations for Training and Validation.
    # NOTE: Heavy augmentation (color jitter, elastic, etc.) is applied OFFLINE
    # during the balancing step. Training transforms are kept light to avoid
    # double-augmenting the pre-augmented minority class images.
    
    return {
        "train": A.Compose([
            A.Resize(img_size, img_size),
            
            # Light geometric augmentations only (for invariance, not diversity)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]),
        
        "val": A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    }