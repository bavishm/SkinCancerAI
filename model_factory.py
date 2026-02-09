import timm

def get_model(model_name, num_classes=7, pretrained=True):
    # Loads a SOTA model directly from the internet.
    print(f"[INFO] Downloading & Loading model: {model_name}...")
    
    try:
        # This will automatically download weights to ~/.cache/torch/hub/checkpoints/
        model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to download {model_name}.")
        print("Check your internet connection or the model name spelling.")
        print(f"Error details: {e}")
        raise e

    return model