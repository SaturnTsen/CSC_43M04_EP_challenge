from dataclasses import dataclass, field
from typing import List, Any, Optional
from multiprocessing import cpu_count
from configs.datamodule.train_transform.clip import TrainTransformConfig
from configs.datamodule.test_transform.clip import TestTransformConfig

@dataclass
class DatasetConfig:
    _target_: str = "data.dataset.Dataset"
    dataset_path: str = "${data_dir}"
    split: str = "train_val"
    metadata: List[str] = field(default_factory=lambda: ["title", "description"])
    transforms: Any = None

@dataclass
class DataModuleConfig:
    _target_: str = "data.datamodule.DataModule"
    dataset_path: str = "${data_dir}"
    train_transform: TrainTransformConfig = field(default_factory=TrainTransformConfig)
    test_transform: TestTransformConfig = field(default_factory=TestTransformConfig)
    metadata: List[str] = field(default_factory=lambda: ["title", "description"])
    batch_size: int = 16  # 进一步减小batch size以适应大型CLIP模型
    num_workers: int = min(cpu_count(), 64)
    seed: int = 42
    val_split: float = 0.1
    standardize_target: bool = False 