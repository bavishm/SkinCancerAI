import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast 
from sklearn.metrics import f1_score, accuracy_score
import os
import time
import requests 
from dotenv import load_dotenv

from dataset_factory import HAM10000Dataset, get_transforms
from model_factory import get_model

# ==========================================
#               CONFIGURATION
# ==========================================
load_dotenv()
WEBHOOK_URL = os.getenv("WEB_HOOK") 

MODEL_NAME = "swinv2_large_window12_192_22k"   # Model A
# MODEL_NAME = "convnext_xlarge.fb_in22k_ft_in1k_384"     # Model B
# MODEL_NAME = "tf_efficientnetv2_l.in21k"     # Model C

IMG_SIZE = 384

if "convnext" in MODEL_NAME:
    BATCH_SIZE = 40
else:    
    BATCH_SIZE = 64

# increase number and add stopping code if validation loss still dropping at epoch 20
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_WORKERS = 16
DATA_DIR = "./data"
IMG_DIR = "./data/all_images"
CHECKPOINT_DIR = "./checkpoints"

def send_curl_log(data):
    # Fixed webhook check to handle None or Placeholder
    if not WEBHOOK_URL or "YOUR_KEY_HERE" in WEBHOOK_URL:
        return 
    try:
        requests.post(WEBHOOK_URL, json=data, timeout=2)
    except Exception:
        pass

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*40)
    print(f"Starting Training: {MODEL_NAME}")
    print(f"GPUs Available:    {torch.cuda.device_count()}")
    print("="*40)
    send_curl_log({"content":f"Currently Training: {MODEL_NAME}", "username":"SCAIInititalization"})

    # Load Data
    train_path = os.path.join(DATA_DIR, "train_split.csv")
    val_path = os.path.join(DATA_DIR, "val_split.csv")
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("Run '01_split_data.py' first!")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Datasets
    transforms = get_transforms(IMG_SIZE)
    train_ds = HAM10000Dataset(train_df, IMG_DIR, transform=transforms['train'])
    val_ds = HAM10000Dataset(val_df, IMG_DIR, transform=transforms['val'])
    
    # Persistent workers for faster data loading (keeps RAM alloc active)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    # Class Weights
    print("Calculating Class Weights...")
    label_map = train_ds.label_map 
    idx_to_class = {v: k for k, v in label_map.items()}
    counts = []
    for i in range(7):
        class_name = idx_to_class[i]
        count = len(train_df[train_df['dx'] == class_name])
        counts.append(count)
    counts = torch.tensor(counts, dtype=torch.float).to(device)
    class_weights = len(train_df) / (7 * counts)
    print(f"  Final Weights: {class_weights.cpu().numpy()}")

    # Model
    model = get_model(MODEL_NAME, num_classes=7)
    if torch.cuda.device_count() > 1:
        print(f"--> Activating DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Optimizer & Scaler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Initialize GradScaler with cuda device type
    scaler = GradScaler('cuda') 

    # [NEW] 1. Initialize History Dictionary
    history = {
        'epoch': [], 
        'train_loss': [], 
        'val_loss': [], 
        'val_acc': [], 
        'val_f1': []
    }

    # Training Loop
    best_val_f1 = 0.0

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        
        current_lr = optimizer.param_groups[0]['lr']
        
        for i, (images, labels) in enumerate(train_loader):
            # ENABLE NON-BLOCKING TRANSFERS HERE
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)

            # Use autocast with device_type='cuda'
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Scale loss and backward
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
            if i % 50 == 0 and i > 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{i}/{len(train_loader)}] Loss: {loss.item():.4f} | LR: {current_lr:.2e}")
                log_payload = {"content": f"Ep {epoch+1} | Step {i} | Loss: {loss.item():.4f} | LR: {current_lr:.2e}","username":"SkinCancerAI"}
                send_curl_log(log_payload)

        epoch_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                # ENABLE NON-BLOCKING TRANSFERS HERE AS WELL
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # autocast in validation too
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    v_loss = criterion(outputs, labels)
                
                val_running_loss += v_loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_running_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        scheduler.step()

        # [NEW] 2. Log Data to History
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        # [NEW] 3. Save CSV immediately (Safe against crashes)
        log_path = os.path.join(CHECKPOINT_DIR, "training_log.csv")
        pd.DataFrame(history).to_csv(log_path, index=False)
        
        duration = time.time() - start_time
        print(f"\n>>> EPOCH {epoch+1} FINISHED ({duration:.1f}s)")
        print(f"    Train Loss: {epoch_loss:.4f}")
        print(f"    Val Loss:   {val_loss:.4f}")
        print(f"    Val Accuracy: {val_acc:.4f}")
        print(f"    Val F1 (Macro): {val_f1:.4f}")
        
        val_payload = {
            "content": f"**EPOCH {epoch+1} DONE**\nTrain Loss: {epoch_loss:.4f}\nVal Loss: {val_loss:.4f}\nVal F1: {val_f1:.4f}"
        }
        send_curl_log(val_payload)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_name = f"best_{MODEL_NAME}.pth"
            save_path = os.path.join(CHECKPOINT_DIR, save_name)
            
            # Save FULL Checkpoint
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_val_f1,
                'config': {'model': MODEL_NAME, 'img_size': IMG_SIZE}
            }
            
            torch.save(checkpoint, save_path)
            print(f"    [SAVED] New Best Checkpoint: {save_path} (F1: {best_val_f1:.4f})")
            send_curl_log({"content": f":rotating_light: **NEW BEST MODEL SAVED!** F1: {best_val_f1:.4f}"})
        
        print("-" * 40)
    
    print(f"Full Training Log saved to: {os.path.join(CHECKPOINT_DIR, 'training_log.csv')}")
    print("TRAINING COMPLETE.")

if __name__ == "__main__":
    main()