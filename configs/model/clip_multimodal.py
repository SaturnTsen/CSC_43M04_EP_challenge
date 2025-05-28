from dataclasses import dataclass, field

@dataclass
class ModelInstanceConfig:
    _target_: str = "models.clip_multimodal.CLIPMultiModal"
    frozen: bool = True
    dropout_rate: float = 0.3
    clip_model_name: str = "openai/clip-vit-large-patch14-336"  # 使用Hugging Face的最大CLIP模型

@dataclass
class ModelConfig:
    instance: ModelInstanceConfig = field(default_factory=ModelInstanceConfig)
    name: str = "CLIP_MULTIMODAL" 