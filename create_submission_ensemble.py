import warnings
warnings.filterwarnings("ignore", message="xFormers is available")

import hydra
from torch.utils.data import DataLoader
import pandas as pd
import torch
import numpy as np
import wandb
from pathlib import Path
from typing import List
from data.dataset import Dataset
from hydra.core.config_store import ConfigStore
from configs.experiments.base import BaseTrainConfig
from utils.transforms import TargetStandardizer


@hydra.main(config_path="configs", config_name=None, version_base="1.3")
def create_submission_ensemble(cfg: BaseTrainConfig):
    """
    创建ensemble提交文件，支持多个fold模型的集成
    """
    
    # 初始化wandb（如果启用）
    logger = None
    if hasattr(cfg, 'log') and cfg.log:
        logger = wandb.init(
            project="challenge_CSC_43M04_EP",
            name=f"{cfg.experiment_name}_submission",
            job_type="ensemble_submission"
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建测试数据加载器
    test_loader = DataLoader(
        Dataset(
            cfg.datamodule.dataset_path,
            "test",
            transforms=hydra.utils.instantiate(cfg.datamodule.test_transform),
            metadata=cfg.datamodule.metadata,
        ),
        batch_size=cfg.datamodule.batch_size,
        shuffle=False,
        num_workers=cfg.datamodule.num_workers,
    )
    
    # 查找所有fold的checkpoint
    checkpoint_dir = Path(cfg.checkpoint_path).parent
    model_name = Path(cfg.checkpoint_path).stem
    
    # 支持两种模式：best和final
    use_best = cfg.get('use_best_checkpoint', True)
    checkpoint_pattern = f"{model_name}_fold*_{'best' if use_best else 'final'}.pt"
    checkpoint_files = list(checkpoint_dir.glob(checkpoint_pattern))
    
    if len(checkpoint_files) == 0:
        raise ValueError(f"No checkpoint files found matching pattern: {checkpoint_pattern}")
    
    print(f"Found {len(checkpoint_files)} checkpoint files:")
    for ckpt in checkpoint_files:
        print(f"  - {ckpt.name}")
    
    # 记录ensemble配置
    if logger is not None:
        logger.log({
            "ensemble/n_models": len(checkpoint_files),
            "ensemble/checkpoint_type": "best" if use_best else "final",
            "ensemble/method": cfg.get('ensemble_method', 'mean'),
            "ensemble/test_samples": len(test_loader.dataset)
        })
    
    # 创建标准化转换器
    target_standardizer = None
    if hasattr(cfg.datamodule, 'standardize_target') and cfg.datamodule.standardize_target:
        target_standardizer = TargetStandardizer(
            mu=cfg.datamodule.target_mu,
            sigma=cfg.datamodule.target_sigma
        )
        print(f"Using target standardizer")
    
    # 存储所有模型的预测
    all_predictions = []
    model_weights_info = []
    
    # 对每个checkpoint进行预测
    for idx, checkpoint_path in enumerate(checkpoint_files):
        print(f"\nProcessing checkpoint {idx + 1}/{len(checkpoint_files)}: {checkpoint_path.name}")
        
        # 加载模型
        model = hydra.utils.instantiate(cfg.model.instance).to(device)
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint)
        model.eval()
        
        # 收集此模型的预测
        fold_predictions = []
        fold_ids = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch["image"] = batch["image"].to(device)
                preds = model(batch).squeeze()
                
                # 如果使用标准化，将预测值转换回原始尺度
                if target_standardizer:
                    preds = target_standardizer.unstandardize(preds)
                
                preds = preds.cpu().numpy()
                fold_predictions.extend(preds)
                fold_ids.extend(batch["id"])
        
        all_predictions.append(np.array(fold_predictions))
        
        # 记录权重信息（如果模型支持）
        if hasattr(model, 'get_interpretability_info'):
            interpretability_info = model.get_interpretability_info()
            if interpretability_info:
                print(f"  Model weights - Image: {interpretability_info['image_weight']:.3f}, "
                      f"Text: {interpretability_info['text_weight']:.3f}, "
                      f"Age: {interpretability_info['age_weight']:.3f}")
                
                model_weights_info.append({
                    'fold': idx,
                    'image_weight': interpretability_info['image_weight'],
                    'text_weight': interpretability_info['text_weight'],
                    'age_weight': interpretability_info['age_weight']
                })
                
                # 记录到wandb
                if logger is not None:
                    logger.log({
                        f"ensemble/model_{idx}/image_weight": interpretability_info['image_weight'],
                        f"ensemble/model_{idx}/text_weight": interpretability_info['text_weight'],
                        f"ensemble/model_{idx}/age_weight": interpretability_info['age_weight']
                    })
    
    # 将所有预测转换为numpy数组
    all_predictions = np.stack(all_predictions, axis=0)  # Shape: [n_models, n_samples]
    
    # 计算ensemble预测
    ensemble_method = cfg.get('ensemble_method', 'mean')
    
    if ensemble_method == 'mean':
        final_predictions = np.mean(all_predictions, axis=0)
        print(f"\nUsing mean ensemble")
    elif ensemble_method == 'median':
        final_predictions = np.median(all_predictions, axis=0)
        print(f"\nUsing median ensemble")
    elif ensemble_method == 'weighted_mean':
        # 可以基于验证性能设置权重
        weights = cfg.get('ensemble_weights', None)
        if weights is None:
            weights = np.ones(len(checkpoint_files)) / len(checkpoint_files)
        else:
            weights = np.array(weights)
            weights = weights / weights.sum()
        
        final_predictions = np.average(all_predictions, axis=0, weights=weights)
        print(f"\nUsing weighted mean ensemble with weights: {weights}")
        
        # 记录权重
        if logger is not None:
            for i, w in enumerate(weights):
                logger.log({f"ensemble/model_{i}_weight": w})
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble_method}")
    
    # 计算预测的统计信息
    pred_std = np.std(all_predictions, axis=0)
    pred_diversity = np.mean(pred_std)  # 模型间预测的多样性
    
    print(f"\nPrediction statistics:")
    print(f"  Mean prediction std: {np.mean(pred_std):.4f}")
    print(f"  Max prediction std: {np.max(pred_std):.4f}")
    print(f"  Min prediction std: {np.min(pred_std):.4f}")
    print(f"  Final prediction range: {np.min(final_predictions):.0f} - {np.max(final_predictions):.0f}")
    
    # 记录统计信息到wandb
    if logger is not None:
        logger.log({
            "ensemble/prediction_diversity": pred_diversity,
            "ensemble/pred_std_mean": np.mean(pred_std),
            "ensemble/pred_std_max": np.max(pred_std),
            "ensemble/pred_std_min": np.min(pred_std),
            "ensemble/final_pred_min": np.min(final_predictions),
            "ensemble/final_pred_max": np.max(final_predictions),
            "ensemble/final_pred_mean": np.mean(final_predictions),
            "ensemble/final_pred_std": np.std(final_predictions)
        })
        
        # 保存权重信息表格
        if model_weights_info:
            weights_df = pd.DataFrame(model_weights_info)
            logger.log({
                "ensemble/model_weights": wandb.Table(dataframe=weights_df)
            })
    
    # 创建提交文件
    records = []
    for id_, pred in zip(fold_ids, final_predictions):
        records.append({"ID": id_.item() if hasattr(id_, 'item') else id_, "views": pred})
    
    submission = pd.DataFrame(records)
    
    # 设置提交文件路径
    submission_path = cfg.submission_path.replace('.csv', f'_ensemble_{ensemble_method}.csv')
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to: {submission_path}")
    
    # 保存ensemble详细信息
    ensemble_info = {
        'method': ensemble_method,
        'n_models': len(checkpoint_files),
        'checkpoints': [str(ckpt) for ckpt in checkpoint_files],
        'mean_std': float(np.mean(pred_std)),
        'prediction_diversity': float(pred_diversity),
        'weights': weights.tolist() if ensemble_method == 'weighted_mean' else None,
        'model_weights': model_weights_info
    }
    
    import json
    info_path = submission_path.replace('.csv', '_info.json')
    with open(info_path, 'w') as f:
        json.dump(ensemble_info, f, indent=2)
    print(f"Ensemble info saved to: {info_path}")
    
    # 记录提交完成
    if logger is not None:
        logger.log({
            "ensemble/submission_created": True,
            "ensemble/submission_path": submission_path
        })
        logger.finish()


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
        # Convert config_name to CamelCase for class name
        # e.g., "clip_multimodal" -> "ClipMultimodal"
        class_name_parts = config_name.split('_')
        class_name = ''.join(word.capitalize() for word in class_name_parts) + 'TrainConfig'
        config_class = getattr(config_module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Cannot load config {config_name}: {str(e)}")
    
    # Register config
    cs = ConfigStore.instance()
    cs.store(name=config_name, node=config_class)
    
    create_submission_ensemble() 