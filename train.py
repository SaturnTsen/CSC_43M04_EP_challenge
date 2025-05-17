import torch
import wandb
import hydra

from tqdm import tqdm
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from hydra.core.config_store import ConfigStore

from data.datamodule import DataModule, BatchDict
from configs.experiments.base import BaseTrainConfig

from utils.sanity import show_images

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
            batch["target"] = batch["target"].to(device).squeeze()
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
                batch["target"] = batch["target"].to(device).squeeze()
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
