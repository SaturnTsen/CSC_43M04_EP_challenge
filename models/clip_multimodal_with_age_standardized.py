import torch
import torch.nn as nn
import open_clip
from typing import List

class CLIPMultiModalWithAgeRegressorStandardized(nn.Module):
    """基于 OpenCLIP 的多模态回归模型，包含年份特征 (标准化版本)

    1. 使用预训练 CLIP backbone 分别编码图像和文本。
    2. 为图像、文本和年份分别设计回归头。
    3. 使用可学习权重对三个预测结果进行加权融合。
    4. 移除最后的 ReLU 激活函数，因为标准化后的目标可能是负数。
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        freeze_backbone: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # 1. 加载预训练 CLIP 模型
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name,
            pretrained=pretrained,
        )
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)

        # 冻结 backbone 参数（可选）
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        embed_dim = self.clip_model.text_projection.shape[1]

        # 2. 图像回归头 (移除最后的 ReLU)
        self.image_regressor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            # 移除 ReLU，允许负值输出
        )

        # 3. 文本回归头 (移除最后的 ReLU)
        self.text_regressor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            # 移除 ReLU，允许负值输出
        )

        # 4. 年份回归头 (移除最后的 ReLU)
        self.age_regressor = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 8, 1),
            # 移除 ReLU，允许负值输出
        )

        # 5. 可学习的融合权重 (使用 softmax 确保权重总和为1)
        # 分别对应 图像、文本、年份 的权重
        self.fusion_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))  # 初始化为相等权重
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化回归头的权重"""
        for regressor in [self.image_regressor, self.text_regressor, self.age_regressor]:
            for m in regressor.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.clip_model.encode_image(image)

    def encode_text(self, text: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(text).to(next(self.clip_model.parameters()).device)
        return self.clip_model.encode_text(tokens)

    def forward(self, batch):
        # batch["image"]: (B, 3, H, W) 已经是 tensor
        # batch["text"]: List[str]
        # batch["age_year"]: (B,) 年份特征
        images = batch["image"]
        texts = batch["text"]
        age_years = batch["age_year"]

        # 1. 编码图像和文本模态
        if self.freeze_backbone:
            with torch.no_grad():
                img_feat = self.encode_image(images)
                text_feat = self.encode_text(texts)
        else:
            img_feat = self.encode_image(images)
            text_feat = self.encode_text(texts)

        # 2. 分别通过各自的回归头
        img_pred = self.image_regressor(img_feat)  # (B, 1)
        text_pred = self.text_regressor(text_feat)  # (B, 1)
        
        # 年份特征需要reshape为 (B, 1) 用于全连接层
        age_input = age_years.float().unsqueeze(1)  # (B, 1)
        age_pred = self.age_regressor(age_input)  # (B, 1)

        # 3. 使用可学习权重进行加权融合
        # 使用 softmax 确保权重总和为1
        weights = torch.softmax(self.fusion_weights, dim=0)
        w_img, w_text, w_age = weights[0], weights[1], weights[2]
        
        fused_pred = w_img * img_pred + w_text * text_pred + w_age * age_pred

        return fused_pred
    
    def get_fusion_weights(self):
        """返回当前的融合权重 (softmax 后的值)"""
        weights = torch.softmax(self.fusion_weights, dim=0)
        return {
            'image_weight': weights[0].item(),
            'text_weight': weights[1].item(),
            'age_weight': weights[2].item()
        }
    
    def get_individual_predictions(self, batch):
        """返回图像、文本和年份的单独预测结果，用于分析"""
        images = batch["image"]
        texts = batch["text"]
        age_years = batch["age_year"]

        # 编码
        if self.freeze_backbone:
            with torch.no_grad():
                img_feat = self.encode_image(images)
                text_feat = self.encode_text(texts)
        else:
            img_feat = self.encode_image(images)
            text_feat = self.encode_text(texts)

        # 分别预测
        img_pred = self.image_regressor(img_feat)
        text_pred = self.text_regressor(text_feat)
        
        age_input = age_years.float().unsqueeze(1)
        age_pred = self.age_regressor(age_input)
        
        return img_pred, text_pred, age_pred 