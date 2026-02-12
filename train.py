import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np  # ADDED
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast 
from sklearn.metrics import f1_score, accuracy_score
import os
import time
import requests 
from dotenv import load_dotenv
from tqdm import tqdm 

from dataset_factory import HAM10000Dataset, get_transforms
from model_factory import get_model

# ==========================================
#               CONFIGURATION
# ==========================================
load_dotenv()
WEBHOOK_URL = os.getenv("WEB_HOOK") 

MODEL_NAME = "swinv2_large_window12to24_192to384_22kft1k" # Model A
# MODEL_NAME = "convnext_xlarge.fb_in22k_ft_in1k_384"     # Model B
IMG_SIZE = 384

if "convnext" in MODEL_NAME:
    BATCH_SIZE = 40
else:    
    BATCH_SIZE = 64

EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_WORKERS = 16
DATA_DIR = "./data"
IMG_DIR = "./data/all_images"
CHECKPOINT_DIR = "./checkpoints"
N_FOLDS = 5 

def send_curl_log(data):
    if not WEBHOOK_URL or "YOUR_KEY_HERE" in WEBHOOK_URL:
        return 
    try:
        requests.post(WEBHOOK_URL, json=data, timeout=2)
    except Exception:
        pass

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available!")
        return
    
    print("="*50)
    print(f"Starting 5-Fold Cross-Validation")
    print(f"Model: {MODEL_NAME}")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Batch Size: {BATCH_SIZE} ({BATCH_SIZE // torch.cuda.device_count()} per GPU)")
    print("="*50)
    
    send_curl_log({"content": f"Starting 5-Fold CV: {MODEL_NAME}"})

    # Load the Master Folds CSV
    folds_path = os.path.join(DATA_DIR, "train_folds.csv")
    if not os.path.exists(folds_path):
        raise FileNotFoundError("Run '01_split_data.py' first!")
    
    full_df = pd.read_csv(folds_path)
    print(f"\nLoaded {len(full_df)} images for CV\n")
    
    # ==========================================
    #           START CROSS-VALIDATION
    # ==========================================
    fold_results = []  # Track results across folds
    
    for fold in range(N_FOLDS):
        print(f"\n{'#'*50}")
        print(f"   FOLD {fold + 1}/{N_FOLDS}")
        print(f"{'#'*50}")
        
        send_curl_log({"content": f"Starting Fold {fold + 1}/{N_FOLDS}"})

        # Dynamic Split
        train_df = full_df[full_df['fold'] != fold].reset_index(drop=True)
        val_df = full_df[full_df['fold'] == fold].reset_index(drop=True)
        
        print(f"Train: {len(train_df)} images | Val: {len(val_df)} images")

        # Datasets & Loaders
        transforms = get_transforms(IMG_SIZE)
        train_ds = HAM10000Dataset(train_df, IMG_DIR, transform=transforms['train'])
        val_ds = HAM10000Dataset(val_df, IMG_DIR, transform=transforms['val'])
        
        train_loader = DataLoader(
            train_ds, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=NUM_WORKERS, 
            pin_memory=True, 
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS, 
            pin_memory=True, 
            persistent_workers=True
        )

        # Class Weights
        print("\nCalculating class weights...")
        label_map = train_ds.label_map 
        idx_to_class = {v: k for k, v in label_map.items()}
        counts = []
        for i in range(7):
            class_name = idx_to_class[i]
            count = len(train_df[train_df['dx'] == class_name])
            counts.append(count)
        counts = torch.tensor(counts, dtype=torch.float).to(device)
        class_weights = len(train_df) / (7 * counts)
        print(f"Weights: {class_weights.cpu().numpy()}\n")

        # Initialize Model (FRESH for every fold)
        model = get_model(MODEL_NAME, num_classes=7)
        if torch.cuda.device_count() > 1:
            print(f"Wrapping with DataParallel ({torch.cuda.device_count()} GPUs)\n")
            model = nn.DataParallel(model)
        model = model.to(device)

        # Optimizer, Scheduler, Scaler
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler = GradScaler('cuda') 

        # History
        history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
        best_fold_f1 = 0.0
        start_epoch = 0

        # ==========================================
        #          RESUME LOGIC
        # ==========================================
        resume_path = os.path.join(CHECKPOINT_DIR, f"last_checkpoint_fold{fold}.pth")
        log_path = os.path.join(CHECKPOINT_DIR, f"log_fold{fold}.csv")

        if os.path.exists(resume_path):
            print(f"Found checkpoint for Fold {fold}. Resuming...")
            checkpoint = torch.load(resume_path, map_location=device)
            
            # Load states
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
                
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint['epoch']
            best_fold_f1 = checkpoint['best_f1']
            
            print(f"Resuming from epoch {start_epoch + 1}/{EPOCHS}")
            print(f"Best F1 so far: {best_fold_f1:.4f}\n")
            
            # Reload history
            if os.path.exists(log_path):
                history = pd.read_csv(log_path).to_dict(orient='list')

        # Check if already done
        if start_epoch >= EPOCHS:
            print(f"Fold {fold} already complete! Skipping.\n")
            # Still track the result
            if os.path.exists(log_path):
                fold_df = pd.read_csv(log_path)
                fold_results.append(fold_df['val_f1'].max())
            continue

        # ==========================
        #       TRAINING LOOP
        # ==========================
        for epoch in range(start_epoch, EPOCHS):
            start_time = time.time()
            model.train()
            running_loss = 0.0
            current_lr = optimizer.param_groups[0]['lr']
            
            train_loop = tqdm(
                train_loader, 
                desc=f"Fold {fold + 1} Epoch {epoch + 1}/{EPOCHS}", 
                leave=False
            )

            for i, (images, labels) in enumerate(train_loop):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)

                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss.item()
                train_loop.set_postfix(loss=loss.item(), lr=f"{current_lr:.2e}")
                
                # Memory check
                if fold == 0 and epoch == 0 and i == 1:
                    tqdm.write(f"\n{'='*50}")
                    tqdm.write("[GPU Memory Check]")
                    for gpu_id in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                        tqdm.write(f"  GPU {gpu_id}: {allocated:.2f}GB / 24GB (reserved: {reserved:.2f}GB)")
                        if allocated > 20:
                            tqdm.write(f"WARNING: Memory usage is high!")
                    tqdm.write(f"{'='*50}\n")

            epoch_loss = running_loss / len(train_loader)

            # Validation
            model.eval()
            val_running_loss = 0.0
            all_preds = []
            all_labels = []
            
            val_loop = tqdm(val_loader, desc="Validation", leave=False)

            with torch.no_grad():
                for images, labels in val_loop:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
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

            # Logging
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(epoch_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            pd.DataFrame(history).to_csv(log_path, index=False)
            
            duration = time.time() - start_time
            print(f"Epoch {epoch + 1}/{EPOCHS} | {duration:.1f}s | "
                  f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val F1: {val_f1:.4f} | LR: {current_lr:.2e}")
            
            # Discord tracking 
            if (epoch + 1) % 5 == 0:
                send_curl_log({
                    "content": f"**Fold {fold + 1} - Epoch {epoch + 1}**\n"
                              f"Val F1: {val_f1:.4f} | Train Loss: {epoch_loss:.4f}"
                })

            # Save best model
            if val_f1 > best_fold_f1:
                best_fold_f1 = val_f1
                save_name = f"best_{MODEL_NAME}_fold{fold}.pth"
                save_path = os.path.join(CHECKPOINT_DIR, save_name)
                
                model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                
                checkpoint = {
                    'fold': fold,
                    'epoch': epoch + 1,
                    'model_state_dict': model_state,
                    'best_f1': best_fold_f1,
                    'val_acc': val_acc,
                    'config': {'model': MODEL_NAME, 'img_size': IMG_SIZE}
                }
                torch.save(checkpoint, save_path)
                print(f"  ✓ Saved best model: {save_name} (F1: {best_fold_f1:.4f})")

            # Save last checkpoint (for resume)
            last_checkpoint = {
                'fold': fold,
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_fold_f1
            }
            torch.save(last_checkpoint, resume_path)
        
        # Cleanup resume checkpoint after successful completion
        if os.path.exists(resume_path):
            os.remove(resume_path)
            print(f"Cleaned up resume checkpoint")
        
        # Track result
        fold_results.append(best_fold_f1)
        
        print(f"\n Fold {fold + 1} complete! Best F1: {best_fold_f1:.4f}")
        send_curl_log({"content": f" **Fold {fold + 1} Complete** | Best F1: {best_fold_f1:.4f}"})
        print("-" * 50)

    # Final summary
    print(f"\n{'='*50}")
    print("CROSS-VALIDATION COMPLETE")
    print(f"{'='*50}")
    
    if len(fold_results) == N_FOLDS:
        mean_f1 = np.mean(fold_results)
        std_f1 = np.std(fold_results)
        print(f"Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"Per-fold F1s: {[f'{f:.4f}' for f in fold_results]}")
        print(f"Min F1: {min(fold_results):.4f}")
        print(f"Max F1: {max(fold_results):.4f}")
        
        send_curl_log({
            "content": f" **ALL FOLDS COMPLETE**\n"
                      f"Mean F1: {mean_f1:.4f} ± {std_f1:.4f}\n"
                      f"Range: [{min(fold_results):.4f}, {max(fold_results):.4f}]"
        })
    else:
        print(f"Warning: Only {len(fold_results)}/{N_FOLDS} folds completed")
    
    print(f"{'='*50}")

if __name__ == "__main__":
    main()