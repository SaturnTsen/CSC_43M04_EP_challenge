from dataclasses import dataclass, field

@dataclass
class ModelInstanceConfig:
    _target_: str = "models.dinov2.DinoV2Finetune"
    frozen: bool = True

@dataclass
class ModelConfig:
    instance: ModelInstanceConfig = field(default_factory=ModelInstanceConfig)
    name: str = "DINOV2"

# 导出配置
ModelConfig = ModelConfig