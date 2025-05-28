from dataclasses import dataclass, field
from configs.datamodule.default import DataModuleConfig

@dataclass 
class DataModuleWithAgeYearConfig(DataModuleConfig):
    """包含age_year特征的数据模块配置"""
    
    # 启用年份特征
    include_age_year: bool = True 