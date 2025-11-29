"""
TeleChat 日志管理模块
提供统一的日志配置和管理
"""

import os
import sys
import logging
from typing import Optional
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

from .config import get_config, LoggingConfig


class TeleChatLogger:
    """TeleChat统一日志管理器"""
    
    _loggers: dict = {}
    _initialized: bool = False
    
    @classmethod
    def setup(cls, config: Optional[LoggingConfig] = None):
        """初始化日志系统"""
        if cls._initialized:
            return
            
        if config is None:
            config = get_config().logging
        
        # 创建日志目录
        if config.file:
            log_dir = Path(config.file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, config.level.upper()))
        
        # 清除现有处理器
        root_logger.handlers.clear()
        
        # 创建格式器
        formatter = logging.Formatter(config.format)
        
        # 控制台处理器
        if config.console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(getattr(logging, config.level.upper()))
            root_logger.addHandler(console_handler)
        
        # 文件处理器
        if config.file:
            file_handler = RotatingFileHandler(
                config.file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(getattr(logging, config.level.upper()))
            root_logger.addHandler(file_handler)
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """获取指定名称的日志器"""
        if not cls._initialized:
            cls.setup()
        
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
        
        return cls._loggers[name]


def get_logger(name: str) -> logging.Logger:
    """获取日志器的便捷函数"""
    return TeleChatLogger.get_logger(name)


def setup_logging(config: Optional[LoggingConfig] = None):
    """设置日志系统"""
    TeleChatLogger.setup(config)


# 预定义的日志器
class LoggerNames:
    """预定义的日志器名称"""
    MAIN = "telechat"
    MODEL = "telechat.model"
    INFERENCE = "telechat.inference"
    API = "telechat.api"
    MEMORY = "telechat.memory"
    TOOLS = "telechat.tools"
    EVOLUTION = "telechat.evolution"
