from dataclasses import dataclass, field

@dataclass
class ModelInstanceConfig:
    _target_: str = "models.dinov2_improved.DinoV2Improved"
    frozen: bool = True
    dropout_rate: float = 0.3

@dataclass
class ModelConfig:
    instance: ModelInstanceConfig = field(default_factory=ModelInstanceConfig)
    name: str = "DINOV2_IMPROVED"

# 导出配置
ModelConfig = ModelConfig 