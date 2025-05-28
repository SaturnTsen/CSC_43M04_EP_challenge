import warnings
warnings.filterwarnings("ignore", message="xFormers is available")

import torch
import wandb
import hydra
import os
import numpy as np
from pathlib import Path

from tqdm import tqdm
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from data.datamodule import DataModule, BatchDict
from data.dataset import Dataset
from configs.experiments.base import BaseTrainConfig
from utils.sanity import show_images
from utils.validation import validate_and_log
from utils.transforms import TargetStandardizer

def train_single_fold(cfg: BaseTrainConfig, fold_idx: int, train_loader: DataLoader, 
                     val_loader: DataLoader, logger=None) -> dict:
    """训练单个折"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    
    # 确定是否使用标准化
    target_standardizer = None
    if hasattr(cfg.datamodule, 'standardize_target') and cfg.datamodule.standardize_target:
        target_standardizer = TargetStandardizer(
            mu=cfg.datamodule.target_mu,
            sigma=cfg.datamodule.target_sigma
        )
        print(f"Fold {fold_idx}: 使用目标标准化 mu={cfg.datamodule.target_mu}, sigma={cfg.datamodule.target_sigma}")
    
    # 创建输出目录
    fold_save_dir = f"{cfg.msle_validation_dir}/fold_{fold_idx}"
    os.makedirs(fold_save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    fold_metrics = {
        'fold': fold_idx,
        'train_losses': [],
        'val_losses': [],
        'msle_scores': []
    }
    
    # 训练循环
    for epoch in tqdm(range(cfg.epochs), desc=f"Fold {fold_idx} Epochs"):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        num_samples_train = 0
        
        pbar = tqdm(train_loader, desc=f"Fold {fold_idx} Epoch {epoch}", leave=False)
        for i, batch in enumerate(pbar):
            batch: BatchDict = batch
            batch["image"] = batch["image"].to(device)
            batch["target"] = batch["target"].to(device).squeeze()
            
            # 如果batch中有age_year特征，也移动到设备上
            if "age_year" in batch:
                batch["age_year"] = batch["age_year"].to(device)
            
            preds = model(batch).squeeze()
            loss = loss_fn(preds, batch["target"])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录融合权重（如果模型支持）
            if hasattr(model, 'get_fusion_weights') and logger is not None:
                fusion_weights = model.get_fusion_weights()
                log_dict = {
                    f"fold_{fold_idx}/train/loss_step": loss.detach().cpu().numpy(),
                    "fold": fold_idx,
                    "epoch": epoch
                }
                # 记录所有融合权重
                for weight_name, weight_value in fusion_weights.items():
                    log_dict[f"fold_{fold_idx}/train/{weight_name}"] = weight_value
                logger.log(log_dict)
            elif hasattr(model, 'get_fusion_weight') and logger is not None:
                # 向后兼容：对于只有两模态的旧模型
                logger.log({
                    f"fold_{fold_idx}/train/fusion_weight": model.get_fusion_weight(),
                    f"fold_{fold_idx}/train/loss_step": loss.detach().cpu().numpy(),
                    "fold": fold_idx,
                    "epoch": epoch
                })
            
            epoch_train_loss += loss.detach().cpu().numpy() * len(batch["image"])
            num_samples_train += len(batch["image"])
            pbar.set_postfix({"loss": loss.detach().cpu().numpy()})
            
        epoch_train_loss /= num_samples_train
        fold_metrics['train_losses'].append(epoch_train_loss)
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        num_samples_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch: BatchDict = batch
                batch["image"] = batch["image"].to(device)
                batch["target"] = batch["target"].to(device).squeeze()
                
                # 如果batch中有age_year特征，也移动到设备上
                if "age_year" in batch:
                    batch["age_year"] = batch["age_year"].to(device)
                
                preds = model(batch).squeeze()
                loss = loss_fn(preds, batch["target"])
                
                epoch_val_loss += loss.detach().cpu().numpy() * len(batch["image"])
                num_samples_val += len(batch["image"])
                
        epoch_val_loss /= num_samples_val
        fold_metrics['val_losses'].append(epoch_val_loss)
        
        # MSLE 验证
        if cfg.msle_validation_interval > 0 and ((epoch + 1) % cfg.msle_validation_interval == 0 or epoch == cfg.epochs - 1):
            msle, _ = validate_and_log(
                model=model,
                val_loader=val_loader,
                device=device,
                epoch=epoch,
                target_standardizer=target_standardizer,
                experiment_name=f"{cfg.experiment_name}_fold_{fold_idx}",
                save_dir=fold_save_dir,
                log_wandb=False,  # 不在这里记录，统一在外面记录
                logger=None
            )
            fold_metrics['msle_scores'].append(msle)
            
            # 记录到 wandb
            if logger is not None:
                logger.log({
                    f"fold_{fold_idx}/train/loss_epoch": epoch_train_loss,
                    f"fold_{fold_idx}/val/loss_epoch": epoch_val_loss,
                    f"fold_{fold_idx}/val/msle": msle,
                    "fold": fold_idx,
                    "epoch": epoch
                })
        else:
            if logger is not None:
                logger.log({
                    f"fold_{fold_idx}/train/loss_epoch": epoch_train_loss,
                    f"fold_{fold_idx}/val/loss_epoch": epoch_val_loss,
                    "fold": fold_idx,
                    "epoch": epoch
                })
        
        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_path = f"checkpoints/{cfg.model.name}_fold_{fold_idx}_best.pt"
            torch.save(model.state_dict(), checkpoint_path)
            fold_metrics['best_checkpoint'] = checkpoint_path
    
    # 保存最终模型
    os.makedirs("checkpoints", exist_ok=True)
    final_checkpoint_path = f"checkpoints/{cfg.model.name}_fold_{fold_idx}_final.pt"
    torch.save(model.state_dict(), final_checkpoint_path)
    fold_metrics['final_checkpoint'] = final_checkpoint_path
    fold_metrics['best_val_loss'] = best_val_loss
    
    return fold_metrics

@hydra.main(config_path="configs", config_name=None, version_base="1.3")
def train_kfold(cfg: BaseTrainConfig) -> None:
    """多模型训练 - 使用固定90/10分割，不同随机种子"""
    
    # 设置训练参数 - 使用90/10分割训练多个模型
    num_models = 5  # 训练5个模型，每个使用不同随机种子
    val_split = 0.1  # 10%验证，90%训练
    
    logger = (
        wandb.init(project="challenge_CSC_43M04_EP", name=f"{cfg.experiment_name}_multi_seed")
        if cfg.log
        else None
    )
    
    # 创建输出目录
    save_dir = cfg.msle_validation_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # 记录配置
    if logger is not None:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        config_dict['num_models'] = num_models
        config_dict['val_split'] = val_split
        logger.config.update(config_dict)
    
    # 存储所有模型的结果
    all_model_metrics = []
    
    # 训练每个模型
    for model_idx in range(num_models):
        print(f"\n{'='*50}")
        print(f"开始训练模型 {model_idx + 1}/{num_models} (种子: {cfg.datamodule.seed + model_idx})")
        print(f"{'='*50}")
        
        # 为每个模型使用不同的随机种子
        current_seed = cfg.datamodule.seed + model_idx
        
        # 创建数据模块（使用不同种子）
        datamodule_config = OmegaConf.to_container(cfg.datamodule, resolve=True)
        datamodule_config['seed'] = current_seed
        datamodule_config['val_split'] = val_split  # 确保使用90/10分割
        base_datamodule = hydra.utils.instantiate(datamodule_config)
        
        # 获取训练和验证加载器
        train_loader = base_datamodule.train_dataloader()
        val_loader = base_datamodule.val_dataloader()
        
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")
        print(f"训练数据比例: {len(train_loader.dataset) / (len(train_loader.dataset) + len(val_loader.dataset)) * 100:.1f}%")
        
        # 训练当前模型
        model_metrics = train_single_fold(
            cfg=cfg,
            fold_idx=model_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            logger=logger
        )
        
        all_model_metrics.append(model_metrics)
        
        print(f"模型 {model_idx} 完成:")
        print(f"  最佳验证损失: {model_metrics['best_val_loss']:.4f}")
        if model_metrics['msle_scores']:
            print(f"  最终 MSLE: {model_metrics['msle_scores'][-1]:.4f}")
    
    # 计算多模型统计
    final_val_losses = [metrics['val_losses'][-1] for metrics in all_model_metrics]
    final_msle_scores = [metrics['msle_scores'][-1] if metrics['msle_scores'] else float('inf') 
                        for metrics in all_model_metrics]
    
    multi_model_val_loss_mean = np.mean(final_val_losses)
    multi_model_val_loss_std = np.std(final_val_losses)
    multi_model_msle_mean = np.mean([score for score in final_msle_scores if score != float('inf')])
    multi_model_msle_std = np.std([score for score in final_msle_scores if score != float('inf')])
    
    print(f"\n{'='*50}")
    print(f"多模型训练结果汇总:")
    print(f"{'='*50}")
    print(f"验证损失: {multi_model_val_loss_mean:.4f} ± {multi_model_val_loss_std:.4f}")
    print(f"MSLE 分数: {multi_model_msle_mean:.4f} ± {multi_model_msle_std:.4f}")
    
    # 记录多模型汇总结果
    if logger is not None:
        logger.log({
            "multi_model_summary/val_loss_mean": multi_model_val_loss_mean,
            "multi_model_summary/val_loss_std": multi_model_val_loss_std,
            "multi_model_summary/msle_mean": multi_model_msle_mean,
            "multi_model_summary/msle_std": multi_model_msle_std,
        })
        
        # 记录每个模型的最终结果
        for i, metrics in enumerate(all_model_metrics):
            logger.log({
                f"multi_model_final/model_{i}_val_loss": metrics['val_losses'][-1],
                f"multi_model_final/model_{i}_msle": metrics['msle_scores'][-1] if metrics['msle_scores'] else None,
            })
    
    # 保存多模型训练结果
    import json
    multi_model_results = {
        'num_models': num_models,
        'val_split': val_split,
        'multi_model_val_loss_mean': multi_model_val_loss_mean,
        'multi_model_val_loss_std': multi_model_val_loss_std,
        'multi_model_msle_mean': multi_model_msle_mean,
        'multi_model_msle_std': multi_model_msle_std,
        'model_metrics': all_model_metrics
    }
    
    with open(f"{save_dir}/multi_model_results.json", 'w') as f:
        json.dump(multi_model_results, f, indent=2)
    
    if cfg.log:
        logger.finish()
    
    print(f"\n多模型训练完成！结果已保存到 {save_dir}/multi_model_results.json")

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
    
    train_kfold() 