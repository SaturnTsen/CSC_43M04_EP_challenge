import torch
import torch.nn as nn
import open_clip
from typing import List

class CLIPMultiModalRegressor(nn.Module):
    """基于 OpenCLIP 的多模态回归模型

    1. 使用预训练 CLIP backbone 分别编码图像和文本。
    2. 为图像和文本分别设计回归头。
    3. 使用可学习权重对两个预测结果进行加权融合。
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

        # 2. 图像回归头
        self.image_regressor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU(),  # 确保非负
        )

        # 3. 文本回归头
        self.text_regressor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU(),  # 确保非负
        )

        # 4. 可学习的融合权重 (使用 sigmoid 确保权重在 [0,1] 范围内)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))  # 初始化为 0.5 (平等权重)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化回归头的权重"""
        for regressor in [self.image_regressor, self.text_regressor]:
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
        images = batch["image"]
        texts = batch["text"]

        # 1. 编码两种模态
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

        # 3. 使用可学习权重进行加权融合
        # alpha 在 [0, 1] 范围内，表示图像预测的权重
        alpha = torch.sigmoid(self.fusion_weight)
        fused_pred = alpha * img_pred + (1 - alpha) * text_pred

        return fused_pred
    
    def get_fusion_weight(self):
        """返回当前的融合权重 (sigmoid 后的值)"""
        return torch.sigmoid(self.fusion_weight).item()
    
    def get_individual_predictions(self, batch):
        """返回图像和文本的单独预测结果，用于分析"""
        images = batch["image"]
        texts = batch["text"]

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
        
        return img_pred, text_pred 