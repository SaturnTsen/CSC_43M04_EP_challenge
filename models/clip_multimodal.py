import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import warnings


class CLIPMultiModal(nn.Module):
    def __init__(self, frozen=True, dropout_rate=0.3, clip_model_name="openai/clip-vit-large-patch14-336"):
        super().__init__()
        # 加载Hugging Face的CLIP模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # 获取CLIP特征维度
        self.feature_dim = self.clip_model.config.projection_dim
        self.frozen = frozen
        
        # 冻结CLIP参数
        if frozen:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        else:
            warnings.warn("Frozen is False, this will train the CLIP parameters")
        
        # 图像回归头
        self.image_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.LayerNorm(self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            nn.LayerNorm(self.feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim // 4, 1)
        )
        
        # 文本回归头（用于title和description的concat特征）
        self.text_head = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.LayerNorm(self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim // 2, 1)
        )
        
        # 年份特征回归头
        self.year_head = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
        
        # 最终融合层
        self.fusion_head = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化回归头的权重"""
        for head in [self.image_head, self.text_head, self.year_head, self.fusion_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def encode_text(self, text_list):
        """编码文本列表"""
        # 处理文本输入
        inputs = self.processor(text=text_list, return_tensors="pt", padding=True, truncation=True, max_length=77)
        
        # 将输入移到正确的设备
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        
        # 获取文本特征
        if self.frozen:
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
        else:
            text_features = self.clip_model.get_text_features(**inputs)
        
        return text_features
    
    def encode_image(self, images):
        """编码图像"""
        # 获取图像特征
        if self.frozen:
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(pixel_values=images)
        else:
            image_features = self.clip_model.get_image_features(pixel_values=images)
        
        return image_features
    
    def forward(self, x):
        # 提取图像特征
        image_features = self.encode_image(x["image"])
        
        # 提取文本特征
        title_features = self.encode_text(x["title"])
        desc_features = self.encode_text(x["description"])
        text_features = torch.cat([title_features, desc_features], dim=1)
        
        # 获取年份特征
        year_features = x["age_years"].unsqueeze(1) if x["age_years"].dim() == 1 else x["age_years"]
        
        # 通过各个回归头
        image_pred = self.image_head(image_features)
        text_pred = self.text_head(text_features)
        year_pred = self.year_head(year_features)
        
        # 融合预测结果
        combined_preds = torch.cat([image_pred, text_pred, year_pred], dim=1)
        final_pred = self.fusion_head(combined_preds)
        
        return final_pred 