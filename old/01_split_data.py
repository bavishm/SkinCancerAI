import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    DATA_DIR = "./data"
    CSV_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
    
    # Check if CSV exists
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Metadata CSV not found at {CSV_PATH}")
    
    # Load Data into df 
    print(f"Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Check if required columns exist
    required_cols = ['lesion_id', 'dx']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Get unique lesions (some lesions are the same just rotated so place them only in t/v/t to prevent cheating)
    unique_lesions = df[['lesion_id', 'dx']].drop_duplicates()
    
    # Display dataset info
    print(f"\nTotal images: {len(df)}")
    print(f"Unique lesions: {df['lesion_id'].nunique()}\n")
    
    print("Class distribution:")
    print(df['dx'].value_counts())
    print()
    
    # 70/15/15 train/val/test split, set aside 30%
    train_ids, temp_ids = train_test_split(
        unique_lesions['lesion_id'], 
        test_size=0.30, 
        stratify=unique_lesions['dx'], 
        random_state=42
    )

    # Get labels for the temp set for stratification
    temp_metadata = unique_lesions[unique_lesions['lesion_id'].isin(temp_ids)]

    # Split 30% non training data into valuation and testing data (15/15)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50, 
        stratify=temp_metadata['dx'],
        random_state=42
    )

    # Create DataFrames
    train_df = df[df['lesion_id'].isin(train_ids)]
    val_df = df[df['lesion_id'].isin(val_ids)]
    test_df = df[df['lesion_id'].isin(test_ids)]

    # Verify no leakage
    assert len(set(train_ids) & set(val_ids)) == 0, "Train/Val leakage detected!"
    assert len(set(train_ids) & set(test_ids)) == 0, "Train/Test leakage detected!"
    assert len(set(val_ids) & set(test_ids)) == 0, "Val/Test leakage detected!"
    print("No data leakage verified\n")

    # Display split info and Save
    total = len(df)
    print(f"Training:   {len(train_df)} images ({len(train_ids)} lesions) (~{len(train_df)/total:.1%})")
    print(f"Validation: {len(val_df)} images ({len(val_ids)} lesions) (~{len(val_df)/total:.1%})")
    print(f"Testing:    {len(test_df)} images ({len(test_ids)} lesions) (~{len(test_df)/total:.1%})")

    train_df.to_csv(os.path.join(DATA_DIR, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test_split.csv"), index=False)
    
    print("\n>>> Split successful (70/15/15). Files saved to", DATA_DIR)

if __name__ == "__main__":
    main()