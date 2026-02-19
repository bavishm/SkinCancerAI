import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np

def main():
    DATA_DIR = "./data"
    CSV_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
    TEST_SIZE = 0.15  # 15% for Final Testing
    N_FOLDS = 5       # 5-Fold CV on the remaining 85%
    SEED = 42
    
    # Load Data
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Metadata CSV not found at {CSV_PATH}")
    
    print(f"Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Group by Lesion ID (avoid leakage)
    unique_lesions = df.drop_duplicates(subset=['lesion_id']).reset_index(drop=True)
    
    print(f"Total Unique Lesions: {len(unique_lesions)}")

    # Create the Test Set
    train_val_lesions, test_lesions = train_test_split(
        unique_lesions, 
        test_size=TEST_SIZE, 
        stratify=unique_lesions['dx'], 
        random_state=SEED
    )
    
    # Create the 5 Folds on the 'train_val_lesions' (The remaining 85%)
    # Initialize fold column
    train_val_lesions = train_val_lesions.copy()
    train_val_lesions["fold"] = -1
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X=train_val_lesions, y=train_val_lesions["dx"])):
        train_val_lesions.iloc[val_idx, train_val_lesions.columns.get_loc("fold")] = fold_idx

    # Map back to original Images
    # We now have lesions assigned to (Test) or (Fold 0, 1, 2, 3, 4)
    # We need to apply this to the full dataframe of images.
    
    # Create Test DataFrame
    test_df = df[df['lesion_id'].isin(test_lesions['lesion_id'])].copy()
    
    # Create Train/CV DataFrame
    # We merge the 'fold' info from our unique_lesions split back to the main image list
    cv_df = df[df['lesion_id'].isin(train_val_lesions['lesion_id'])].copy()
    cv_df = cv_df.merge(train_val_lesions[['lesion_id', 'fold']], on='lesion_id', how='left')
    
    # Safety Checks
    # Ensure no overlap
    assert len(set(test_df['lesion_id']) & set(cv_df['lesion_id'])) == 0, "DATA LEAKAGE DETECTED!"
    
    # Save Files
    test_path = os.path.join(DATA_DIR, "test_split.csv")
    cv_path = os.path.join(DATA_DIR, "train_folds.csv")
    
    test_df.to_csv(test_path, index=False)
    cv_df.to_csv(cv_path, index=False)
    
    print("\n" + "="*40)
    print("SPLIT SUMMARY")
    print("="*40)
    print(f"Total Images: {len(df)}")
    print(f"Held-Out Test Set (15%): {len(test_df)} images (Saved to {test_path})")
    print(f"Cross-Validation Set (85%): {len(cv_df)} images (Saved to {cv_path})")
    print("-" * 20)
    print("Fold Distribution in CV Set:")
    print(cv_df['fold'].value_counts().sort_index())
    print("\n Ready for training!")

if __name__ == "__main__":
    main()