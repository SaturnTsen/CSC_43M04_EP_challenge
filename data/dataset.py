import torch
import pandas as pd
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, metadata, transforms=None, include_age_year=True):
        self.dataset_path = dataset_path
        self.split = split
        self.include_age_year = include_age_year
        
        # - read the info csvs
        print(f"Reading {dataset_path}/{split}.csv")
        info = pd.read_csv(f"{dataset_path}/{split}.csv")
        info["description"] = info["description"].fillna("")
        info["meta"] = info[metadata].agg(" + ".join, axis=1)
        if "views" in info.columns:
            self.targets = info["views"].values

        # - ids
        self.ids = info["id"].values
        # - text
        self.text = info["meta"].values
        
        # - age_year feature: 计算 2024 - year
        if self.include_age_year and "year" in info.columns:
            # 确保年份数据有效，填充缺失值
            info["year"] = info["year"].fillna(2020)  # 默认值设为2020
            self.age_years = 2024 - info["year"].values
            print(f"Age year feature enabled. Range: [{self.age_years.min()}, {self.age_years.max()}]")
        else:
            self.age_years = None

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
            "text": self.text[idx],
        }
        
        # - add age_year feature if enabled
        if self.include_age_year and self.age_years is not None:
            value["age_year"] = torch.tensor(self.age_years[idx], dtype=torch.float32)
        
        # - don't have the target for test
        if hasattr(self, "targets"):
            value["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return value
