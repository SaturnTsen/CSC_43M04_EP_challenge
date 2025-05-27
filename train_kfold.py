import warnings
warnings.filterwarnings("ignore", message="xFormers is available")

import torch
import wandb
import hydra
import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold

from tqdm import tqdm
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from data.datamodule import DataModule, BatchDict
from data.dataset import Dataset
from configs.experiments.base import BaseTrainConfig
from utils.sanity import show_images
from utils.validation import validate_and_log
from utils.transforms import TargetStandardizer

class KFoldDataModule:
    """支持 K 折交叉验证的数据模块"""
    
    def __init__(self, base_datamodule: DataModule, k_folds: int = 5, seed: int = 42):
        self.base_datamodule = base_datamodule
        self.k_folds = k_folds
        self.seed = seed
        
        # 获取完整数据集
        self.full_dataset = base_datamodule.train_val_dataset
        
        # 创建 K 折划分
        self.kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        self.fold_splits = list(self.kfold.split(range(len(self.full_dataset))))
        
        print(f"创建 {k_folds} 折交叉验证，总数据量: {len(self.full_dataset)}")
        
    def get_fold_dataloaders(self, fold_idx: int):
        """获取指定折的训练和验证数据加载器"""
        if fold_idx >= self.k_folds:
            raise ValueError(f"fold_idx {fold_idx} 超出范围 [0, {self.k_folds-1}]")
            
        train_indices, val_indices = self.fold_splits[fold_idx]
        
        print(f"Fold {fold_idx}: 训练集 {len(train_indices)} 样本, 验证集 {len(val_indices)} 样本")
        
        # 创建子集
        train_subset = Subset(self.full_dataset, train_indices)
        val_subset = Subset(self.full_dataset, val_indices)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_subset,
            batch_size=self.base_datamodule.batch_size,
            shuffle=True,
            num_workers=self.base_datamodule.num_workers,
            collate_fn=self.base_datamodule.collate_fn,
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.base_datamodule.batch_size,
            shuffle=False,
            num_workers=self.base_datamodule.num_workers,
            collate_fn=self.base_datamodule.collate_fn,
        )
        
        return train_loader, val_loader

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
            
            preds = model(batch).squeeze()
            loss = loss_fn(preds, batch["target"])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录融合权重（如果模型支持）
            if hasattr(model, 'get_fusion_weight') and logger is not None:
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
    """K 折交叉验证训练"""
    
    # 设置 K 折参数
    k_folds = 5  # 可以通过配置文件设置
    
    logger = (
        wandb.init(project="challenge_CSC_43M04_EP", name=f"{cfg.experiment_name}_kfold")
        if cfg.log
        else None
    )
    
    # 创建基础数据模块
    base_datamodule = hydra.utils.instantiate(cfg.datamodule)
    
    # 创建 K 折数据模块
    kfold_datamodule = KFoldDataModule(base_datamodule, k_folds=k_folds, seed=cfg.datamodule.seed)
    
    # 创建输出目录
    save_dir = cfg.msle_validation_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # 记录配置
    if logger is not None:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        config_dict['k_folds'] = k_folds
        logger.config.update(config_dict)
    
    # 存储所有折的结果
    all_fold_metrics = []
    
    # 训练每一折
    for fold_idx in range(k_folds):
        print(f"\n{'='*50}")
        print(f"开始训练 Fold {fold_idx + 1}/{k_folds}")
        print(f"{'='*50}")
        
        # 获取当前折的数据加载器
        train_loader, val_loader = kfold_datamodule.get_fold_dataloaders(fold_idx)
        
        # 训练当前折
        fold_metrics = train_single_fold(
            cfg=cfg,
            fold_idx=fold_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            logger=logger
        )
        
        all_fold_metrics.append(fold_metrics)
        
        print(f"Fold {fold_idx} 完成:")
        print(f"  最佳验证损失: {fold_metrics['best_val_loss']:.4f}")
        if fold_metrics['msle_scores']:
            print(f"  最终 MSLE: {fold_metrics['msle_scores'][-1]:.4f}")
    
    # 计算交叉验证统计
    final_val_losses = [metrics['val_losses'][-1] for metrics in all_fold_metrics]
    final_msle_scores = [metrics['msle_scores'][-1] if metrics['msle_scores'] else float('inf') 
                        for metrics in all_fold_metrics]
    
    cv_val_loss_mean = np.mean(final_val_losses)
    cv_val_loss_std = np.std(final_val_losses)
    cv_msle_mean = np.mean([score for score in final_msle_scores if score != float('inf')])
    cv_msle_std = np.std([score for score in final_msle_scores if score != float('inf')])
    
    print(f"\n{'='*50}")
    print(f"交叉验证结果汇总:")
    print(f"{'='*50}")
    print(f"验证损失: {cv_val_loss_mean:.4f} ± {cv_val_loss_std:.4f}")
    print(f"MSLE 分数: {cv_msle_mean:.4f} ± {cv_msle_std:.4f}")
    
    # 记录交叉验证汇总结果
    if logger is not None:
        logger.log({
            "cv_summary/val_loss_mean": cv_val_loss_mean,
            "cv_summary/val_loss_std": cv_val_loss_std,
            "cv_summary/msle_mean": cv_msle_mean,
            "cv_summary/msle_std": cv_msle_std,
        })
        
        # 记录每折的最终结果
        for i, metrics in enumerate(all_fold_metrics):
            logger.log({
                f"cv_final/fold_{i}_val_loss": metrics['val_losses'][-1],
                f"cv_final/fold_{i}_msle": metrics['msle_scores'][-1] if metrics['msle_scores'] else None,
            })
    
    # 保存交叉验证结果
    import json
    cv_results = {
        'k_folds': k_folds,
        'cv_val_loss_mean': cv_val_loss_mean,
        'cv_val_loss_std': cv_val_loss_std,
        'cv_msle_mean': cv_msle_mean,
        'cv_msle_std': cv_msle_std,
        'fold_metrics': all_fold_metrics
    }
    
    with open(f"{save_dir}/cv_results.json", 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    if cfg.log:
        logger.finish()
    
    print(f"\n交叉验证完成！结果已保存到 {save_dir}/cv_results.json")

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