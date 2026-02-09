import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class HAM10000Dataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
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
        
        return image, torch.tensor(label, dtype=torch.long)

def get_transforms(img_size=384):
    
    # Returns a dictionary of transformations for Training and Validation.
    # Targeting 384x384 for SwinV2/EfficientNet on 3090s.
    
    return {
        "train": A.Compose([
            A.Resize(img_size, img_size),
            
            # Geometric Augmentations (Crucial for invariance)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.2),
            
            # Color/Texture Augmentations (Simulate different lighting/cameras)
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5)),
                A.GaussNoise(var_limit=(10.0, 50.0)),
            ], p=0.2),

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]),
        
        "val": A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    }