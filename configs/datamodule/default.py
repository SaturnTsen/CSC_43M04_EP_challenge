from dataclasses import dataclass, field
from typing import List, Any, Optional
from multiprocessing import cpu_count
from configs.datamodule.train_transform.simple import TrainTransformConfig
from configs.datamodule.test_transform.simple import TestTransformConfig

@dataclass
class DatasetConfig:
    _target_: str = "data.dataset.Dataset"
    dataset_path: str = "${data_dir}"
    split: str = "train_val"
    metadata: List[str] = field(default_factory=lambda: ["title"])
    transforms: Any = None

@dataclass
class DataModuleConfig:
    _target_: str = "data.datamodule.DataModule"
    dataset_path: str = "${data_dir}"
    train_transform: TrainTransformConfig = field(default_factory=TrainTransformConfig)
    test_transform: TestTransformConfig = field(default_factory=TestTransformConfig)
    metadata: List[str] = field(default_factory=lambda: ["title"])
    batch_size: int = 1024
    num_workers: int = min(cpu_count(), 64)  # 根据CPU核心数设置，最大64
    seed: int = 114514
    val_split: float = 0.1
    standardize_target: bool = False