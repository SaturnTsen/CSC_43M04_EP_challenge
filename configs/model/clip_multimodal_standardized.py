from dataclasses import dataclass, field

@dataclass
class ModelInstanceConfig:
    _target_: str = "models.clip_multimodal_standardized.CLIPMultiModalRegressorStandardized"
    # CLIP backbone 名称，例如 "ViT-B-32", "ViT-L-14" 等
    clip_model_name: str = "ViT-B-32"
    pretrained: str = "openai"  # openai / laion2b_s34b_b79k 等
    freeze_backbone: bool = True
    hidden_dim: int = 512
    dropout: float = 0.3

@dataclass
class ModelConfig:
    instance: ModelInstanceConfig = field(default_factory=ModelInstanceConfig)
    name: str = "CLIP_MM_STANDARDIZED" 