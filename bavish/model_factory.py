import timm
import torch
import torch.nn as nn
from dataset_factory import METADATA_DIM

class FiLMModel(nn.Module):
    """Multi-stage FiLM (Feature-wise Linear Modulation) wrapper for timm backbones.
    
    Applies FiLM conditioning at EVERY stage of the backbone, plus the final
    pooled features. This lets patient metadata influence feature extraction
    at multiple levels of abstraction:
      - Early stages: low-level texture/color modulation
      - Middle stages: structural pattern modulation
      - Late stages: high-level semantic modulation
      - Final pooled: global feature modulation before classification
    
    Supports ConvNeXt (NCHW) and SwinV2 (NHWC) architectures.
    """
    def __init__(self, backbone, feature_dim, metadata_dim=METADATA_DIM, num_classes=7,
                 drop_rate=0.3, model_name=''):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        
        # Detect architecture type
        if hasattr(backbone, 'stages'):       # ConvNeXt family
            self.arch_type = 'convnext'
            self.data_format = 'NCHW'
            num_stages = len(backbone.stages)
        elif hasattr(backbone, 'layers'):     # Swin / SwinV2 family
            self.arch_type = 'swin'
            self.data_format = 'NHWC'
            num_stages = len(backbone.layers)
        else:
            raise ValueError("Unsupported backbone for multi-stage FiLM (need 'stages' or 'layers')")
        
        # Get per-stage channel dimensions
        stage_dims = self._detect_stage_dims(model_name, num_stages)
        print(f"[INFO] Multi-stage FiLM: arch={self.arch_type}, stage_dims={stage_dims}")
        
        # --- Per-stage spatial FiLM generators ---
        self.stage_films = nn.ModuleList()
        for dim in stage_dims:
            film_gen = nn.Sequential(
                nn.Linear(metadata_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_rate),
                nn.Linear(128, dim * 2)       # gamma and beta concatenated
            )
            # Initialize to identity (gamma=1 after +1 offset, beta=0)
            nn.init.zeros_(film_gen[-1].weight)
            nn.init.zeros_(film_gen[-1].bias)
            self.stage_films.append(film_gen)
        
        # --- Final FiLM on pooled 1-D features ---
        self.film_generator = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(128, feature_dim * 2)
        )
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feature_dim, num_classes)
        )
    
    def _detect_stage_dims(self, model_name, num_stages):
        """Get channel dimensions for each backbone stage."""
        # Try backbone's feature_info (available on most timm models)
        if hasattr(self.backbone, 'feature_info'):
            fi = self.backbone.feature_info
            if hasattr(fi, 'channels'):
                dims = fi.channels()
            elif isinstance(fi, (list, tuple)):
                dims = [info['num_chs'] for info in fi]
            else:
                dims = None
            if dims and len(dims) == num_stages:
                return dims
        # Fallback: create a lightweight temp model to query feature_info
        if model_name:
            temp = timm.create_model(model_name, pretrained=False, features_only=True)
            dims = temp.feature_info.channels()
            del temp
            if len(dims) >= num_stages:
                return dims[:num_stages]
        raise ValueError(f"Could not detect stage channel dimensions for {model_name}")
    
    def _apply_spatial_film(self, x, metadata, stage_idx):
        """Apply spatial FiLM modulation to an intermediate feature map."""
        film_params = self.stage_films[stage_idx](metadata)   # [B, dim*2]
        gamma, beta = film_params.chunk(2, dim=-1)            # each [B, dim]
        gamma = gamma + 1.0                                   # identity offset
        
        if x.dim() == 4:                          # spatial feature map
            if self.data_format == 'NCHW':         # [B, C, H, W]
                gamma = gamma[:, :, None, None]
                beta  = beta[:, :, None, None]
            else:                                  # [B, H, W, C]
                gamma = gamma[:, None, None, :]
                beta  = beta[:, None, None, :]
        elif x.dim() == 3:                         # sequence [B, L, C]
            gamma = gamma[:, None, :]
            beta  = beta[:, None, :]
        
        return gamma * x + beta
    
    def forward(self, images, metadata=None):
        # ---- Stem / Patch embed ----
        if self.arch_type == 'convnext':
            x = self.backbone.stem(images)
            stages = self.backbone.stages
        else:  # swin
            x = self.backbone.patch_embed(images)
            if hasattr(self.backbone, 'absolute_pos_embed') and self.backbone.absolute_pos_embed is not None:
                x = x + self.backbone.absolute_pos_embed
            if hasattr(self.backbone, 'pos_drop') and self.backbone.pos_drop is not None:
                x = self.backbone.pos_drop(x)
            stages = self.backbone.layers
        
        # ---- Stages with per-stage FiLM ----
        for i, stage in enumerate(stages):
            x = stage(x)
            if metadata is not None:
                x = self._apply_spatial_film(x, metadata, i)
        
        # ---- Pre-head normalization ----
        if self.arch_type == 'convnext':
            x = self.backbone.norm_pre(x)
        else:  # swin
            x = self.backbone.norm(x)
        
        # ---- Global pool → 1-D features ----
        features = self.backbone.forward_head(x)              # [B, feature_dim]
        
        # ---- Final FiLM on pooled features ----
        if metadata is not None:
            film_params = self.film_generator(metadata)
            gamma, beta = film_params.chunk(2, dim=-1)
            gamma = gamma + 1.0
            features = gamma * features + beta
        
        return self.classifier(features)


def get_model(model_name, num_classes=7, pretrained=True, drop_rate=0.3, drop_path_rate=0.2, use_film=False):
    """Load a timm model, optionally wrapped with FiLM conditioning.
    
    Args:
        model_name: timm model name
        num_classes: number of output classes
        pretrained: use ImageNet pretrained weights
        drop_rate: dropout rate
        drop_path_rate: stochastic depth rate inside the backbone
        use_film: if True, wrap with FiLM metadata conditioning
    
    Returns:
        model: nn.Module (either plain timm model or FiLMModel wrapper)
    """
    film_str = " + FiLM" if use_film else ""
    print(f"[INFO] Loading model: {model_name}{film_str} (drop_rate={drop_rate}, drop_path_rate={drop_path_rate})...")
    
    try:
        if use_film:
            # Load backbone WITHOUT classifier (num_classes=0 removes the head)
            backbone = timm.create_model(
                model_name, 
                pretrained=pretrained, 
                num_classes=0,  # Removes classifier, returns pooled features
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
            )
            feature_dim = backbone.num_features  # e.g., 2048 for ConvNeXt-XLarge, 1536 for SwinV2-Large
            print(f"[INFO] Backbone feature dim: {feature_dim}, Metadata dim: {METADATA_DIM}")
            
            model = FiLMModel(
                backbone=backbone,
                feature_dim=feature_dim,
                metadata_dim=METADATA_DIM,
                num_classes=num_classes,
                drop_rate=drop_rate,
                model_name=model_name
            )
        else:
            model = timm.create_model(
                model_name, 
                pretrained=pretrained, 
                num_classes=num_classes,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
            )
    except Exception as e:
        print(f"\n[ERROR] Failed to load {model_name}.")
        print(f"Error details: {e}")
        raise e

    return model