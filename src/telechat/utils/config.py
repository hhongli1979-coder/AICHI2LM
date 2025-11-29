"""
TeleChat 配置管理模块
支持YAML配置文件和环境变量
"""

import os
import yaml
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """模型配置"""
    name: str = "TeleChat-12B"
    path: str = "./models/12B"
    dtype: str = "float16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    use_cache: bool = True


@dataclass
class InferenceConfig:
    """推理配置"""
    max_length: int = 4096
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


@dataclass
class ServiceConfig:
    """服务配置"""
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    timeout: int = 300
    reload: bool = False
    cors_origins: list = field(default_factory=lambda: ["*"])


@dataclass
class MemoryConfig:
    """记忆系统配置"""
    enabled: bool = True
    database_url: str = "sqlite:///telechat_memory.db"
    max_history: int = 20
    short_term_capacity: int = 100
    long_term_enabled: bool = True


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = "logs/telechat.log"
    console: bool = True


@dataclass
class TeleChatConfig:
    """TeleChat总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "TeleChatConfig":
        """从YAML文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TeleChatConfig":
        """从字典创建配置"""
        config = cls()
        
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'inference' in config_dict:
            config.inference = InferenceConfig(**config_dict['inference'])
        if 'service' in config_dict:
            config.service = ServiceConfig(**config_dict['service'])
        if 'memory' in config_dict:
            config.memory = MemoryConfig(**config_dict['memory'])
        if 'logging' in config_dict:
            config.logging = LoggingConfig(**config_dict['logging'])
            
        return config
    
    @classmethod
    def from_env(cls) -> "TeleChatConfig":
        """从环境变量加载配置"""
        config = cls()
        
        # 模型配置
        if os.getenv('TELECHAT_MODEL_PATH'):
            config.model.path = os.getenv('TELECHAT_MODEL_PATH')
        if os.getenv('TELECHAT_MODEL_DTYPE'):
            config.model.dtype = os.getenv('TELECHAT_MODEL_DTYPE')
            
        # 服务配置
        if os.getenv('TELECHAT_HOST'):
            config.service.host = os.getenv('TELECHAT_HOST')
        if os.getenv('TELECHAT_PORT'):
            config.service.port = int(os.getenv('TELECHAT_PORT'))
            
        # 推理配置
        if os.getenv('TELECHAT_MAX_LENGTH'):
            config.inference.max_length = int(os.getenv('TELECHAT_MAX_LENGTH'))
        if os.getenv('TELECHAT_TEMPERATURE'):
            config.inference.temperature = float(os.getenv('TELECHAT_TEMPERATURE'))
            
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model': self.model.__dict__,
            'inference': self.inference.__dict__,
            'service': self.service.__dict__,
            'memory': self.memory.__dict__,
            'logging': self.logging.__dict__
        }
    
    def save_yaml(self, config_path: str):
        """保存为YAML文件"""
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)


# 全局配置实例
_global_config: Optional[TeleChatConfig] = None


def get_config() -> TeleChatConfig:
    """获取全局配置"""
    global _global_config
    if _global_config is None:
        _global_config = TeleChatConfig()
    return _global_config


def load_config(config_path: str = None) -> TeleChatConfig:
    """加载配置"""
    global _global_config
    
    if config_path and os.path.exists(config_path):
        _global_config = TeleChatConfig.from_yaml(config_path)
    else:
        _global_config = TeleChatConfig.from_env()
    
    return _global_config
