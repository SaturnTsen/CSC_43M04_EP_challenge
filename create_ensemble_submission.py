import warnings
warnings.filterwarnings("ignore", message="xFormers is available")

import hydra
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import json

from data.dataset import Dataset
from hydra.core.config_store import ConfigStore
from configs.experiments.base import BaseTrainConfig
from utils.transforms import TargetStandardizer

@hydra.main(config_path="configs", config_name=None, version_base="1.3")
def create_ensemble_submission(cfg: BaseTrainConfig):
    """使用 K 折训练的多个模型创建集成预测"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 首先创建一个临时的 datamodule 来获取 collate_fn
    temp_datamodule = hydra.utils.instantiate(cfg.datamodule)
    
    # 创建测试数据加载器
    test_loader = DataLoader(
        Dataset(
            cfg.datamodule.dataset_path,
            "test",
            transforms=hydra.utils.instantiate(cfg.datamodule.test_transform),
            metadata=cfg.datamodule.metadata,
            include_age_year=getattr(cfg.datamodule, 'include_age_year', False),  # 添加age_year支持
        ),
        batch_size=cfg.datamodule.batch_size,
        shuffle=False,
        num_workers=cfg.datamodule.num_workers,
        collate_fn=temp_datamodule.collate_fn,  # 使用正确的 collate_fn
    )
    
    # 创建标准化转换器（如果启用）
    target_standardizer = None
    if hasattr(cfg.datamodule, 'standardize_target') and cfg.datamodule.standardize_target:
        target_standardizer = TargetStandardizer(
            mu=cfg.datamodule.target_mu,
            sigma=cfg.datamodule.target_sigma
        )
        print(f"使用目标标准化: mu={cfg.datamodule.target_mu}, sigma={cfg.datamodule.target_sigma}")
    
    # 查找所有模型的检查点
    num_models = 5  # 与训练时保持一致，使用5个模型
    model_checkpoints = []
    
    for model_idx in range(num_models):
        # 优先使用最佳模型，如果不存在则使用最终模型
        best_checkpoint = f"checkpoints/{cfg.model.name}_fold_{model_idx}_best.pt"  # 保持文件名格式兼容
        final_checkpoint = f"checkpoints/{cfg.model.name}_fold_{model_idx}_final.pt"
        
        if Path(best_checkpoint).exists():
            model_checkpoints.append(best_checkpoint)
            print(f"模型 {model_idx}: 使用最佳模型 {best_checkpoint}")
        elif Path(final_checkpoint).exists():
            model_checkpoints.append(final_checkpoint)
            print(f"模型 {model_idx}: 使用最终模型 {final_checkpoint}")
        else:
            print(f"警告: 模型 {model_idx} 的模型文件不存在，跳过")
    
    if not model_checkpoints:
        raise FileNotFoundError("没有找到任何模型检查点文件！请先运行多模型训练。")
    
    print(f"找到 {len(model_checkpoints)} 个模型，开始集成预测...")
    
    # 加载所有模型
    models = []
    for checkpoint_path in model_checkpoints:
        model = hydra.utils.instantiate(cfg.model.instance).to(device)
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint)
        model.eval()
        models.append(model)
        print(f"加载模型: {checkpoint_path}")
    
    # 进行集成预测
    all_predictions = []
    all_ids = []
    
    print("开始预测...")
    with torch.no_grad():
        for batch in test_loader:
            batch["image"] = batch["image"].to(device)
            
            # 如果batch中有age_year特征，也移动到设备上
            if "age_year" in batch:
                batch["age_year"] = batch["age_year"].to(device)
            
            batch_ids = batch["id"]
            
            # 收集所有模型的预测
            batch_predictions = []
            for model in models:
                preds = model(batch).squeeze()
                
                # 如果使用标准化，将预测值转换回原始尺度
                if target_standardizer:
                    preds = target_standardizer.unstandardize(preds)
                
                batch_predictions.append(preds.cpu().numpy())
            
            # 计算集成预测（平均）
            ensemble_preds = np.mean(batch_predictions, axis=0)
            
            all_predictions.extend(ensemble_preds)
            all_ids.extend(batch_ids)
    
    # 创建提交文件
    records = []
    for id_, pred in zip(all_ids, all_predictions):
        records.append({"ID": id_, "views": pred})
    
    submission = pd.DataFrame(records)
    
    # 生成集成提交文件名
    ensemble_submission_path = cfg.submission_path.replace('.csv', '_ensemble.csv')
    submission.to_csv(ensemble_submission_path, index=False)
    
    print(f"集成提交文件已保存到: {ensemble_submission_path}")
    print(f"预测统计:")
    print(f"  样本数量: {len(submission)}")
    print(f"  预测值范围: [{submission['views'].min():.2f}, {submission['views'].max():.2f}]")
    print(f"  预测值均值: {submission['views'].mean():.2f}")
    print(f"  预测值中位数: {submission['views'].median():.2f}")
    
    # 保存集成预测的详细信息
    ensemble_info = {
        'num_models': len(models),
        'model_checkpoints': model_checkpoints,
        'ensemble_method': 'mean',
        'standardized': target_standardizer is not None,
        'prediction_stats': {
            'count': len(submission),
            'min': float(submission['views'].min()),
            'max': float(submission['views'].max()),
            'mean': float(submission['views'].mean()),
            'median': float(submission['views'].median()),
            'std': float(submission['views'].std())
        }
    }
    
    ensemble_info_path = ensemble_submission_path.replace('.csv', '_info.json')
    with open(ensemble_info_path, 'w') as f:
        json.dump(ensemble_info, f, indent=2)
    
    print(f"集成信息已保存到: {ensemble_info_path}")
    
    return ensemble_submission_path

if __name__ == "__main__":
    import importlib
    import sys
    
    # Get config-name from command line arguments
    config_name = None
    for arg in sys.argv:
        if arg.startswith("--config-name="):
            config_name = arg.split("=")[1]
            break
    
    if config_name is None:
        raise ValueError("Please use --config-name to specify the config name")
    
    # Dynamically import config class
    try:
        config_module = importlib.import_module(f"configs.experiments.{config_name}")
        config_class = getattr(config_module, f"{config_name.capitalize()}TrainConfig")
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Cannot load config {config_name}: {str(e)}")
    
    # Register config
    cs = ConfigStore.instance()
    cs.store(name=config_name, node=config_class)
    
    create_ensemble_submission() 