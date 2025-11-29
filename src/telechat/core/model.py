"""
TeleChat 模型管理模块
负责模型的加载、管理和生命周期
"""

import os
import gc
from typing import Optional, Dict, Any, Union
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)

from ..utils.config import get_config, ModelConfig
from ..utils.logging import get_logger
from ..utils.exceptions import ModelException, ErrorCodes

logger = get_logger("telechat.model")


class TeleChatModel:
    """TeleChat模型管理器"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        初始化模型管理器
        
        Args:
            config: 模型配置，如果为None则使用全局配置
        """
        self.config = config or get_config().model
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.generation_config: Optional[GenerationConfig] = None
        self._loaded = False
        
    @property
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._loaded and self.model is not None
    
    def load(self, model_path: Optional[str] = None) -> "TeleChatModel":
        """
        加载模型
        
        Args:
            model_path: 模型路径，如果为None则使用配置中的路径
            
        Returns:
            self: 返回自身以支持链式调用
        """
        path = model_path or self.config.path
        
        if not os.path.exists(path):
            raise ModelException(
                message=f"模型路径不存在: {path}",
                model_name=self.config.name,
                code=ErrorCodes.MODEL_NOT_FOUND
            )
        
        logger.info(f"开始加载模型: {path}")
        
        try:
            # 确定数据类型
            dtype = self._get_torch_dtype()
            
            # 加载分词器
            logger.info("加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                path,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # 加载模型
            logger.info("加载模型权重...")
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                trust_remote_code=self.config.trust_remote_code,
                device_map=self.config.device_map,
                torch_dtype=dtype
            )
            
            # 加载生成配置
            logger.info("加载生成配置...")
            self.generation_config = GenerationConfig.from_pretrained(path)
            
            # 设置为评估模式
            self.model.eval()
            
            self._loaded = True
            logger.info(f"模型加载完成: {self.config.name}")
            
            return self
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise ModelException(
                message=f"模型加载失败: {str(e)}",
                model_name=self.config.name,
                code=ErrorCodes.MODEL_LOAD_ERROR,
                cause=e
            )
    
    def unload(self):
        """卸载模型，释放资源"""
        logger.info("卸载模型...")
        
        if self.model is not None:
            del self.model
            self.model = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        self.generation_config = None
        self._loaded = False
        
        # 清理GPU缓存
        self._cleanup_gpu()
        
        logger.info("模型已卸载")
    
    def chat(
        self,
        question: str,
        history: Optional[list] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, tuple]:
        """
        对话接口
        
        Args:
            question: 用户问题
            history: 对话历史
            stream: 是否流式输出
            **kwargs: 其他生成参数
            
        Returns:
            如果stream=False，返回(answer, history)
            如果stream=True，返回生成器
        """
        if not self.is_loaded:
            raise ModelException(
                message="模型未加载",
                code=ErrorCodes.MODEL_LOAD_ERROR
            )
        
        history = history or []
        
        try:
            result = self.model.chat(
                tokenizer=self.tokenizer,
                question=question,
                history=history,
                generation_config=self.generation_config,
                stream=stream,
                **kwargs
            )
            return result
            
        except Exception as e:
            logger.error(f"推理失败: {str(e)}")
            raise ModelException(
                message=f"推理失败: {str(e)}",
                code=ErrorCodes.MODEL_INFERENCE_ERROR,
                cause=e
            )
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        **kwargs
    ) -> str:
        """
        文本生成接口
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            **kwargs: 其他生成参数
            
        Returns:
            生成的文本
        """
        if not self.is_loaded:
            raise ModelException(
                message="模型未加载",
                code=ErrorCodes.MODEL_LOAD_ERROR
            )
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    generation_config=self.generation_config,
                    **kwargs
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text
            
        except Exception as e:
            logger.error(f"生成失败: {str(e)}")
            raise ModelException(
                message=f"生成失败: {str(e)}",
                code=ErrorCodes.MODEL_INFERENCE_ERROR,
                cause=e
            )
    
    def _get_torch_dtype(self) -> torch.dtype:
        """获取PyTorch数据类型"""
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16
        }
        return dtype_map.get(self.config.dtype.lower(), torch.float16)
    
    def _cleanup_gpu(self):
        """清理GPU缓存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            "name": self.config.name,
            "path": self.config.path,
            "loaded": self._loaded,
            "dtype": self.config.dtype,
            "device_map": self.config.device_map
        }
        
        if self._loaded and self.model is not None:
            info["num_parameters"] = sum(p.numel() for p in self.model.parameters())
            info["device"] = str(next(self.model.parameters()).device)
            
        return info
    
    def __enter__(self):
        """上下文管理器入口"""
        if not self._loaded:
            self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.unload()
        return False


# 全局模型实例
_global_model: Optional[TeleChatModel] = None


def get_model() -> TeleChatModel:
    """获取全局模型实例"""
    global _global_model
    if _global_model is None:
        _global_model = TeleChatModel()
    return _global_model


def load_model(model_path: Optional[str] = None) -> TeleChatModel:
    """加载全局模型"""
    model = get_model()
    if not model.is_loaded:
        model.load(model_path)
    return model
