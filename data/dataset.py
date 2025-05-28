import torch
import pandas as pd
from PIL import Image
from datetime import datetime


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, metadata, transforms=None):
        self.dataset_path = dataset_path
        self.split = split
        # - read the info csvs
        print(f"Reading {dataset_path}/{split}.csv")
        info = pd.read_csv(f"{dataset_path}/{split}.csv")
        info["description"] = info["description"].fillna("")
        info["title"] = info["title"].fillna("")
        
        # 计算年龄（当前年份 - 上传年份）
        current_year = datetime.now().year
        info["age_year"] = current_year - info["year"]
        
        # 组合文本信息用于CLIP
        # 使用title和description的组合
        info["text"] = info["title"] + " " + info["description"]
        
        info["meta"] = info[metadata].agg(" + ".join, axis=1)
        if "views" in info.columns:
            self.targets = info["views"].values

        # - ids
        self.ids = info["id"].values
        # - text (for CLIP)
        self.text = info["text"].values
        # - title
        self.titles = info["title"].values
        # - description
        self.descriptions = info["description"].values
        # - age in years
        self.age_years = info["age_year"].values
        # - meta (legacy compatibility)
        self.meta = info["meta"].values

        # - transforms
        self.transforms = transforms

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        # - load the image
        image = Image.open(
            f"{self.dataset_path}/{self.split}/{self.ids[idx]}.jpg"
        ).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        value = {
            "id": self.ids[idx],
            "image": image,
            "text": self.text[idx],  # Combined text for CLIP
            "title": self.titles[idx],
            "description": self.descriptions[idx],
            "age_year": torch.tensor(self.age_years[idx], dtype=torch.float32),
            "meta": self.meta[idx]  # Keep for compatibility
        }
        # - don't have the target for test
        if hasattr(self, "targets"):
            value["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return value
