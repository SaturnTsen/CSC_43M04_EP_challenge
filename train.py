import warnings
warnings.filterwarnings("ignore", message="xFormers is available")

import torch
import wandb
import hydra
import os
from pathlib import Path

from tqdm import tqdm
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from data.datamodule import DataModule, BatchDict
from configs.experiments.base import BaseTrainConfig
from utils.sanity import show_images, check_initial_loss
from utils.validation import validate_and_log  # 导入验证函数
from utils.transforms import TargetStandardizer  # 导入标准化工具

@hydra.main(config_path="configs", config_name=None, version_base="1.3")
def train(cfg: BaseTrainConfig) -> None:
    logger = (
        wandb.init(project="challenge_CSC_43M04_EP", name=cfg.experiment_name)
        if cfg.log
        else None
    )
    
    device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model : nn.Module = hydra.utils.instantiate(cfg.model.instance).to(device)
    
    optimizer : Optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn : nn.Module = hydra.utils.instantiate(cfg.loss_fn)
    
    datamodule : DataModule = hydra.utils.instantiate(cfg.datamodule)
    train_loader : DataLoader = datamodule.train_dataloader()
    val_loader : DataLoader = datamodule.val_dataloader()
    
    # 确定是否使用标准化
    target_standardizer = None
    if hasattr(cfg.datamodule, 'standardize_target') and cfg.datamodule.standardize_target:
        target_standardizer = TargetStandardizer(
            mu=cfg.datamodule.target_mu,
            sigma=cfg.datamodule.target_sigma
        )
        print(f"Use target standardization: mu={cfg.datamodule.target_mu}, sigma={cfg.datamodule.target_sigma}")
    
    train_sanity : wandb.Image = show_images(train_loader, name="assets/sanity/train_images")
    (
        logger.log({"sanity_checks/train_images": wandb.Image(train_sanity)})
        if logger is not None
        else None
    )
    if val_loader is not None:
        val_sanity = show_images(val_loader, name="assets/sanity/val_images")
        logger.log(
            {"sanity_checks/val_images": wandb.Image(val_sanity)}
        ) if logger is not None else None

    # 创建输出目录 - 使用Hydra的工作目录
    save_dir = cfg.msle_validation_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"MSLE validation results will be saved to: {save_dir}")

    # 获取msle验证频率
    msle_validation_interval = cfg.msle_validation_interval
    
    # 🔍 检查初始损失（训练前）
    print("\n" + "="*60)
    print("🔍 训练前初始损失检查")
    print("="*60)
    
    initial_train_loss = check_initial_loss(
        model=model, 
        data_loader=train_loader, 
        loss_fn=loss_fn, 
        device=device, 
        name="训练集"
    )
    
    if val_loader is not None and len(val_loader) > 0:
        initial_val_loss = check_initial_loss(
            model=model, 
            data_loader=val_loader, 
            loss_fn=loss_fn, 
            device=device, 
            name="验证集"
        )
    else:
        initial_val_loss = float('inf')
        print("⚠️  验证集为空，跳过验证集初始损失检查")
    
    # 记录初始损失到wandb
    if logger is not None:
        logger.log({
            "initial_loss/train": initial_train_loss,
            "initial_loss/val": initial_val_loss,
        })
    
    print(f"\n📊 初始损失总结:")
    print(f"  训练集: {initial_train_loss:.4f}")
    print(f"  验证集: {initial_val_loss:.4f}")
    print("="*60)
    
    # 记录训练配置
    if logger is not None:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        logger.config.update(config_dict)
        logger.log({
            "validation/msle_interval": msle_validation_interval,
            "validation/save_dir": save_dir
        })

    # -- loop over epochs
    for epoch in tqdm(range(cfg.epochs), desc="Epochs"):
        # -- loop over training batches
        model.train()
        epoch_train_loss = 0
        num_samples_train = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for i, batch in enumerate(pbar):
            batch: BatchDict = batch
            batch["image"] = batch["image"].to(device)
            # text 字段作为字符串列表保持在 CPU 上，模型会在内部进行 tokenization
            batch["target"] = batch["target"].to(device).squeeze()
            
            # 如果batch中有age_year特征，也移动到设备上
            if "age_year" in batch:
                batch["age_year"] = batch["age_year"].to(device)
                
            preds = model(batch).squeeze()
            loss = loss_fn(preds, batch["target"])
            (
                logger.log({"loss": loss.detach().cpu().numpy()})
                if logger is not None
                else None
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 如果损失函数支持lambda scheduler，则更新lambda值
            if hasattr(loss_fn, 'step'):
                current_lambda = loss_fn.step()
                if logger is not None and hasattr(loss_fn, 'get_current_lambda'):
                    logger.log({"train/current_lambda": loss_fn.get_current_lambda()})
            
            # 记录融合权重（如果模型支持）
            if hasattr(model, 'get_fusion_weights') and logger is not None:
                fusion_weights = model.get_fusion_weights()
                # 记录所有融合权重
                for weight_name, weight_value in fusion_weights.items():
                    logger.log({f"train/{weight_name}": weight_value})
            elif hasattr(model, 'get_fusion_weight') and logger is not None:
                # 向后兼容：对于只有两模态的旧模型
                logger.log({"train/fusion_weight": model.get_fusion_weight()})
            
            epoch_train_loss += loss.detach().cpu().numpy() * len(batch["image"])
            num_samples_train += len(batch["image"])
            pbar.set_postfix({"train/loss_step": loss.detach().cpu().numpy()})
        epoch_train_loss /= num_samples_train
        (
            logger.log(
                {
                    "epoch": epoch,
                    "train/loss_epoch": epoch_train_loss,
                }
            )
            if logger is not None
            else None
        )

        # -- validation loop
        val_metrics = {}
        epoch_val_loss = 0
        num_samples_val = 0
        model.eval()
        if val_loader is not None: 
            for _, batch in enumerate(val_loader):
                batch: BatchDict = batch
                batch["image"] = batch["image"].to(device)
                # text 字段作为字符串列表保持在 CPU 上
                batch["target"] = batch["target"].to(device).squeeze()
                
                # 如果batch中有age_year特征，也移动到设备上
                if "age_year" in batch:
                    batch["age_year"] = batch["age_year"].to(device)
                    
                with torch.no_grad():
                    preds = model(batch).squeeze()
                loss = loss_fn(preds, batch["target"])
                epoch_val_loss += loss.detach().cpu().numpy() * len(batch["image"])
                num_samples_val += len(batch["image"])
            epoch_val_loss /= num_samples_val
            val_metrics["val/loss_epoch"] = epoch_val_loss
            (
                logger.log(
                    {
                        "epoch": epoch,
                        **val_metrics,
                    }
                )
                if logger is not None
                else None
            )
            
        # 根据配置的间隔进行MSLE验证
        if msle_validation_interval > 0 and ((epoch + 1) % msle_validation_interval == 0 or epoch == cfg.epochs - 1):
            print(f"\nExecuting MSLE validation for epoch {epoch}...")
            # 运行验证并记录MSLE和预测结果
            msle, _ = validate_and_log(
                model=model,
                val_loader=val_loader,
                device=device,
                epoch=epoch,
                target_standardizer=target_standardizer,
                experiment_name=cfg.experiment_name,
                save_dir=save_dir,
                log_wandb=cfg.log,
                logger=logger
            )
            # 在wandb中记录MSLE (使用统一的命名)
            if logger is not None:
                logger.log({
                    "epoch": epoch,
                    "validation/msle": msle
                })

    print(
        f"""Epoch {epoch}: 
        Training metrics:
        - Train Loss: {epoch_train_loss:.4f},
        Validation metrics: 
        - Val Loss: {epoch_val_loss:.4f}"""
    )

    if cfg.log:
        logger.finish()

    torch.save(model.state_dict(), cfg.checkpoint_path)


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
    
    train()
