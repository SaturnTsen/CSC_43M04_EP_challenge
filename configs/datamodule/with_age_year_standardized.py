from dataclasses import dataclass, field
from configs.datamodule.standardized import StandardizedDataModuleConfig

@dataclass 
class DataModuleWithAgeYearStandardizedConfig(StandardizedDataModuleConfig):
    """包含age_year特征和标准化的数据模块配置"""
    
    # 启用年份特征
    include_age_year: bool = True 