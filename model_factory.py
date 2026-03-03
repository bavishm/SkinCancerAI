import timm
import torch
import torch.nn as nn
from dataset_factory import METADATA_DIM

class FiLMModel(nn.Module):
    """FiLM (Feature-wise Linear Modulation) wrapper for any timm backbone.
    
    Takes a pretrained timm model and adds metadata conditioning:
    1. Backbone extracts pooled image features (e.g., 2048-d for ConvNeXt-XLarge)
    2. FiLM generator converts metadata → (gamma, beta) modulation parameters
    3. Features are modulated: output = gamma * features + beta
    4. Modulated features are classified by a new head
    
    This lets patient metadata (age, sex, localization) influence the image features
    without the risk of token domination that cross-attention would have.
    """
    def __init__(self, backbone, feature_dim, metadata_dim=METADATA_DIM, num_classes=7, drop_rate=0.3):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        
        # FiLM generator: metadata → (gamma, beta) for feature modulation
        self.film_generator = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(128, feature_dim * 2)  # gamma and beta concatenated
        )
        
        # Initialize FiLM to identity transform (gamma=1, beta=0)
        # This means at the start of training, FiLM has no effect — pure image model
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)
        # Set gamma bias to 0 (after +1 offset in forward, effective gamma=1)
        
        # New classifier head (replaces the backbone's original head)
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, images, metadata=None):
        # Extract pooled features from backbone (classifier removed)
        features = self.backbone(images)  # [B, feature_dim]
        
        if metadata is not None:
            # Generate FiLM parameters from metadata
            film_params = self.film_generator(metadata)  # [B, feature_dim * 2]
            gamma, beta = film_params.chunk(2, dim=-1)   # each [B, feature_dim]
            gamma = gamma + 1.0  # Offset so initial gamma ≈ 1 (identity)
            
            # Apply FiLM: feature-wise affine transformation
            features = gamma * features + beta
        
        return self.classifier(features)


def get_model(model_name, num_classes=7, pretrained=True, drop_rate=0.3, use_film=False):
    """Load a timm model, optionally wrapped with FiLM conditioning.
    
    Args:
        model_name: timm model name
        num_classes: number of output classes
        pretrained: use ImageNet pretrained weights
        drop_rate: dropout rate
        use_film: if True, wrap with FiLM metadata conditioning
    
    Returns:
        model: nn.Module (either plain timm model or FiLMModel wrapper)
    """
    film_str = " + FiLM" if use_film else ""
    print(f"[INFO] Loading model: {model_name}{film_str} (drop_rate={drop_rate})...")
    
    try:
        if use_film:
            # Load backbone WITHOUT classifier (num_classes=0 removes the head)
            backbone = timm.create_model(
                model_name, 
                pretrained=pretrained, 
                num_classes=0,  # Removes classifier, returns pooled features
                drop_rate=drop_rate
            )
            feature_dim = backbone.num_features  # e.g., 2048 for ConvNeXt-XLarge, 1536 for SwinV2-Large
            print(f"[INFO] Backbone feature dim: {feature_dim}, Metadata dim: {METADATA_DIM}")
            
            model = FiLMModel(
                backbone=backbone,
                feature_dim=feature_dim,
                metadata_dim=METADATA_DIM,
                num_classes=num_classes,
                drop_rate=drop_rate
            )
        else:
            model = timm.create_model(
                model_name, 
                pretrained=pretrained, 
                num_classes=num_classes,
                drop_rate=drop_rate
            )
    except Exception as e:
        print(f"\n[ERROR] Failed to load {model_name}.")
        print(f"Error details: {e}")
        raise e

    return model