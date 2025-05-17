from dataclasses import dataclass, field
from typing import List, Any
from configs.datamodule.train_transform.simple import TrainTransformConfig
from configs.datamodule.test_transform.simple import TestTransformConfig

@dataclass
class DataModuleConfig:
    _target_: str = "data.datamodule.DataModule"
    dataset_path: str = "${data_dir}"
    train_transform: TrainTransformConfig = field(default_factory=TrainTransformConfig)
    test_transform: TestTransformConfig = field(default_factory=TestTransformConfig)
    metadata: List[str] = field(default_factory=lambda: ["title"])
    batch_size: int = 128
    num_workers: int = 64
    seed: int = 42