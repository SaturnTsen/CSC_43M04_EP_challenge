import warnings
warnings.filterwarnings("ignore", message="xFormers is available")

import torch
import wandb
import hydra
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from torch.multiprocessing import spawn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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


def setup_ddp(rank, world_size):
    """设置分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()


def train_fold(rank, world_size, cfg: BaseTrainConfig, fold_idx: int, train_indices: np.ndarray, val_indices: np.ndarray):
    """训练单个fold"""
    # 设置分布式训练
    if world_size > 1:
        setup_ddp(rank, world_size)
    
    # 设置设备
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # 创建fold特定的实验名称
    fold_experiment_name = f"{cfg.experiment_name}_fold{fold_idx}"
    
    # 初始化wandb（只在主进程）
    logger = None
    if rank == 0 and cfg.log:
        logger = wandb.init(
            project="challenge_CSC_43M04_EP", 
            name=fold_experiment_name,
            group=cfg.experiment_name,
            job_type=f"fold_{fold_idx}"
        )
    
    # 创建模型
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    
    # 如果使用多GPU，包装为DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # 创建优化器和损失函数
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    
    # 创建数据集
    full_dataset = Dataset(
        cfg.datamodule.dataset_path,
        "train_val",
        transforms=hydra.utils.instantiate(cfg.datamodule.train_transform),
        metadata=cfg.datamodule.metadata,
    )
    
    # 创建训练和验证子集
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    
    # 创建数据加载器
    train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    val_sampler = DistributedSampler(val_subset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.datamodule.batch_size,
        shuffle=(train_sampler is None),
        num_workers=cfg.datamodule.num_workers,
        sampler=train_sampler,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.datamodule.batch_size,
        shuffle=False,
        num_workers=cfg.datamodule.num_workers,
        sampler=val_sampler,
        pin_memory=True
    )
    
    # 设置标准化器
    target_standardizer = None
    if hasattr(cfg.datamodule, 'standardize_target') and cfg.datamodule.standardize_target:
        target_standardizer = TargetStandardizer(
            mu=cfg.datamodule.target_mu,
            sigma=cfg.datamodule.target_sigma
        )
        if rank == 0:
            print(f"Fold {fold_idx}: Using target standardization")
    
    # 创建保存目录
    save_dir = os.path.join(cfg.msle_validation_dir, f"fold_{fold_idx}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 记录配置（只在主进程）
    if rank == 0 and logger is not None:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        config_dict['fold'] = fold_idx
        config_dict['train_size'] = len(train_indices)
        config_dict['val_size'] = len(val_indices)
        logger.config.update(config_dict)
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in tqdm(range(cfg.epochs), desc=f"Fold {fold_idx} Epochs", disable=(rank != 0)):
        # 设置epoch（对于DistributedSampler）
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        num_samples_train = 0
        
        pbar = tqdm(train_loader, desc=f"Fold {fold_idx} Epoch {epoch}", leave=False, disable=(rank != 0))
        for i, batch in enumerate(pbar):
            batch: BatchDict = batch
            batch["image"] = batch["image"].to(device)
            batch["target"] = batch["target"].to(device).squeeze()
            
            preds = model(batch).squeeze()
            loss = loss_fn(preds, batch["target"])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.detach().cpu().numpy() * len(batch["image"])
            num_samples_train += len(batch["image"])
            
            if rank == 0:
                pbar.set_postfix({"train/loss_step": loss.detach().cpu().numpy()})
                if logger is not None and i % 10 == 0:
                    logger.log({
                        f"fold_{fold_idx}/train/loss_step": loss.detach().cpu().numpy(),
                        "epoch": epoch,
                        "step": epoch * len(train_loader) + i
                    })
        
        epoch_train_loss /= num_samples_train
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        num_samples_val = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch: BatchDict = batch
                batch["image"] = batch["image"].to(device)
                batch["target"] = batch["target"].to(device).squeeze()
                
                preds = model(batch).squeeze()
                loss = loss_fn(preds, batch["target"])
                
                epoch_val_loss += loss.detach().cpu().numpy() * len(batch["image"])
                num_samples_val += len(batch["image"])
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch["target"].cpu().numpy())
        
        epoch_val_loss /= num_samples_val
        
        # 记录权重信息（如果模型支持）
        if rank == 0 and hasattr(model, 'module') and hasattr(model.module, 'get_interpretability_info'):
            interpretability_info = model.module.get_interpretability_info()
            if interpretability_info and logger is not None:
                logger.log({
                    f"fold_{fold_idx}/weights/image": interpretability_info['image_weight'],
                    f"fold_{fold_idx}/weights/text": interpretability_info['text_weight'],
                    f"fold_{fold_idx}/weights/age": interpretability_info['age_weight'],
                    "epoch": epoch
                })
        
        # 计算MSLE
        if rank == 0:
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            
            # 如果使用了标准化，先反标准化
            if target_standardizer:
                all_preds = target_standardizer.unstandardize(torch.tensor(all_preds)).numpy()
                all_targets = target_standardizer.unstandardize(torch.tensor(all_targets)).numpy()
            
            # 计算MSLE
            msle = np.mean((np.log1p(all_preds) - np.log1p(all_targets)) ** 2)
            
            if logger is not None:
                logger.log({
                    f"fold_{fold_idx}/epoch": epoch,
                    f"fold_{fold_idx}/train/loss_epoch": epoch_train_loss,
                    f"fold_{fold_idx}/val/loss_epoch": epoch_val_loss,
                    f"fold_{fold_idx}/val/msle": msle
                })
            
            print(f"Fold {fold_idx} Epoch {epoch}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, MSLE: {msle:.4f}")
            
            # 保存最佳模型
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                checkpoint_path = os.path.join(cfg.checkpoint_path.replace('.pt', f'_fold{fold_idx}_best.pt'))
                torch.save(model.state_dict() if not isinstance(model, DDP) else model.module.state_dict(), 
                          checkpoint_path)
    
    # 保存最终模型
    if rank == 0:
        final_checkpoint_path = os.path.join(cfg.checkpoint_path.replace('.pt', f'_fold{fold_idx}_final.pt'))
        torch.save(model.state_dict() if not isinstance(model, DDP) else model.module.state_dict(), 
                  final_checkpoint_path)
        
        if logger is not None:
            logger.finish()
    
    # 清理
    if world_size > 1:
        cleanup()


@hydra.main(config_path="configs", config_name=None, version_base="1.3")
def train_kfold(cfg: BaseTrainConfig) -> None:
    """主函数：执行3-fold交叉验证训练"""
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 读取数据以获取索引
    info = pd.read_csv(f"{cfg.datamodule.dataset_path}/train_val.csv")
    n_samples = len(info)
    indices = np.arange(n_samples)
    
    # 创建3-fold分割
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    # 确定可用的GPU数量
    n_gpus = torch.cuda.device_count()
    world_size = min(n_gpus, 3)  # 最多使用3个GPU
    
    print(f"Starting 3-fold cross-validation training")
    print(f"Total samples: {n_samples}")
    print(f"Using {world_size} GPU(s)")
    
    # 保存fold信息
    fold_info = []
    
    # 训练每个fold
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(indices)):
        print(f"\nTraining Fold {fold_idx + 1}/3")
        print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")
        
        fold_info.append({
            'fold': fold_idx,
            'train_size': len(train_indices),
            'val_size': len(val_indices)
        })
        
        if world_size > 1 and fold_idx < world_size:
            # 并行训练（如果有多个GPU）
            spawn(train_fold, 
                  args=(world_size, cfg, fold_idx, train_indices, val_indices),
                  nprocs=1,  # 每次只启动一个进程
                  join=True)
        else:
            # 单GPU训练
            train_fold(0, 1, cfg, fold_idx, train_indices, val_indices)
    
    print("\nAll folds training completed!")
    
    # 保存fold信息
    fold_df = pd.DataFrame(fold_info)
    fold_df.to_csv(os.path.join(cfg.root_dir, 'fold_info.csv'), index=False)


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
    
    train_kfold() 