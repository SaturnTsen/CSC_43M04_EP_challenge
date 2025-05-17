import torch
import torch.nn as nn
import warnings

class DinoV2ImprovedStandardized(nn.Module):
    def __init__(self, frozen=False, dropout_rate=0.3):
        super().__init__()
        # 加载预训练的DinoV2骨干网络
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()
        self.dim = self.backbone.norm.normalized_shape[0]
        
        # 冻结骨干网络参数（如果需要）
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            warnings.warn("Frozen is False, this will train the backbone parameters")
        
        # 改进的回归头设计
        self.regression_head = nn.Sequential(
            nn.Linear(self.dim, self.dim // 2),
            nn.LayerNorm(self.dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(self.dim // 2, self.dim // 4),
            nn.LayerNorm(self.dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(self.dim // 4, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化回归头的权重"""
        for m in self.regression_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 提取特征
        features = self.backbone(x["image"])
        # 通过回归头
        output = self.regression_head(features)
        return output 