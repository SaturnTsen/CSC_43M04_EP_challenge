from dataclasses import dataclass, field
from typing import List
from configs.dataset.train_transform.simple import TrainTransformConfig
from configs.dataset.val_transform.simple import ValTransformConfig

@dataclass
class DatasetConfig:
    batch_size: int = 128
    num_workers: int = 16
    seed: int = 42
    metadata: List[str] = field(default_factory=lambda: ["title"])
    train_transform: TrainTransformConfig = field(default_factory=TrainTransformConfig)
    val_transform: ValTransformConfig = field(default_factory=ValTransformConfig)

# 导出配置
DatasetConfig = DatasetConfig