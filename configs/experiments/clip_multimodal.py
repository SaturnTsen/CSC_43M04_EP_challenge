from dataclasses import dataclass, field
from typing import List, Any, Optional
from configs.model.clip_multimodal import ModelConfig
from configs.optim.adamw import OptimizerConfig
from configs.loss_fn.msle import LossFnConfig
from configs.datamodule.default import DataModuleConfig, DatasetConfig
from configs.experiments.base import BaseTrainConfig


@dataclass
class ClipMultimodalTrainConfig(BaseTrainConfig):
    """CLIP多模态模型的训练配置"""
    
    # 训练参数
    epochs: int = 30  # 增加训练轮数
    log: bool = True
    
    # 模型配置
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # 优化器配置 - 为CLIP调整学习率
    optim: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(
        lr=1e-4,  # CLIP通常需要较小的学习率
        weight_decay=1e-2
    ))
    
    # 数据模块配置 - 调整批次大小以适应大型模型
    datamodule: DataModuleConfig = field(default_factory=lambda: DataModuleConfig(
        batch_size=16,  # 根据3090显存调整
        num_workers=4,
        metadata=["title", "description"],  # 使用的元数据字段
        standardize_target=True,  # 使用目标标准化
        target_mu=10.5,  # 根据数据集调整
        target_sigma=2.0
    ))
    
    # MSLE验证频率
    msle_validation_interval: int = 5
    
    # Ensemble配置
    use_best_checkpoint: bool = True
    ensemble_method: str = "mean"  # mean, median, weighted_mean
    ensemble_weights: Optional[List[float]] = None  # 如果使用weighted_mean 