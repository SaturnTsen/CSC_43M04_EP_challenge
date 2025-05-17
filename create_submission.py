import warnings
warnings.filterwarnings("ignore", message="xFormers is available")

import hydra
from torch.utils.data import DataLoader
import pandas as pd
import torch
from data.dataset import Dataset
from hydra.core.config_store import ConfigStore
from configs.experiments.base import BaseTrainConfig

@hydra.main(config_path="configs", config_name=None, version_base="1.3")
def create_submission(cfg: BaseTrainConfig):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    # - Load model and checkpoint
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    checkpoint = torch.load(cfg.checkpoint_path, weights_only=True)
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    model.load_state_dict(checkpoint)
    print("Model loaded")

    # - Create records
    records = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch["image"] = batch["image"].to(device)
            preds = model(batch).squeeze().cpu().numpy()
        
            for id_, pred in zip(batch["id"], preds):
                records.append({"ID": id_.item(), "views": pred})

    # - Create submission.csv
    submission = pd.DataFrame(records)
    submission.to_csv(cfg.submission_path, index=False)


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
    
    create_submission()
