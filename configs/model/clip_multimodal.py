from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class CLIPMultimodalInstance:
    _target_: str = "models.clip_multimodal.CLIPMultimodalModel"
    clip_model: str = "ViT-L/14"  # 使用大型CLIP模型，适合3090
    hidden_dim: int = 512
    dropout: float = 0.1
    freeze_clip: bool = True # 是否冻结CLIP参数


@dataclass
class ModelConfig:
    name: str = "clip_multimodal"
    instance: CLIPMultimodalInstance = field(default_factory=CLIPMultimodalInstance) 