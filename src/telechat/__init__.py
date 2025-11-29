# TeleChat Core Package
"""
TeleChat - 星辰语义大模型核心包
"""

__version__ = "2.0.0"
__author__ = "TeleAI"

from .core.model import TeleChatModel
from .core.inference import InferenceEngine

__all__ = [
    "TeleChatModel",
    "InferenceEngine",
    "__version__"
]
