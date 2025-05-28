from dataclasses import dataclass, field

from configs.experiments.base import BaseTrainConfig
from configs.model.clip_multimodal_with_age import ModelConfig
from configs.optim.adamw import OptimizerConfig
from configs.loss_fn.msle import LossFnConfig
from configs.datamodule.with_age_year import DataModuleWithAgeYearConfig


@dataclass
class Clip_mm_with_ageTrainConfig(BaseTrainConfig):
    """使用 CLIP 多模态模型 + 年份特征的训练配置"""

    # 实验名称
    experiment_name: str = "clip_mm_with_age_experiment"

    # 模型配置 (使用包含年份特征的 CLIP 多模态)
    model: ModelConfig = field(default_factory=ModelConfig)

    # 优化器配置
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)

    # 损失函数配置 (使用MSLE)
    loss_fn: LossFnConfig = field(default_factory=LossFnConfig)

    # 数据模块配置 (使用包含年份特征的DataModule)
    datamodule: DataModuleWithAgeYearConfig = field(default_factory=DataModuleWithAgeYearConfig)

    # 训练轮数
    epochs: int = 30

    # 检查点 & 提交路径
    checkpoint_path: str = "checkpoints/clip_mm_with_age.pt"
    submission_path: str = "submissions/clip_mm_with_age.csv"

    # 记录到 wandb
    log: bool = True

    # MSLE 验证间隔
    msle_validation_interval: int = 3
    msle_validation_dir: str = "${hydra:run.dir}/validations" 