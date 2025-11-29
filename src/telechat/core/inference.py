"""
TeleChat 推理引擎模块
提供高性能的推理服务
"""

import time
import asyncio
from typing import Optional, List, Dict, Any, Generator, AsyncGenerator
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

import torch

from .model import TeleChatModel, get_model
from ..utils.config import get_config, InferenceConfig
from ..utils.logging import get_logger
from ..utils.exceptions import ModelException, ErrorCodes

logger = get_logger("telechat.inference")


@dataclass
class InferenceRequest:
    """推理请求"""
    prompt: str
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    stream: bool = False
    request_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
            "stream": self.stream,
            "request_id": self.request_id
        }


@dataclass
class InferenceResponse:
    """推理响应"""
    text: str
    request_id: Optional[str] = None
    tokens_generated: int = 0
    time_taken: float = 0.0
    finish_reason: str = "stop"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "request_id": self.request_id,
            "tokens_generated": self.tokens_generated,
            "time_taken": self.time_taken,
            "finish_reason": self.finish_reason
        }


@dataclass 
class ChatMessage:
    """对话消息"""
    role: str  # "user" or "bot"
    content: str
    timestamp: Optional[str] = None


class InferenceEngine:
    """推理引擎"""
    
    def __init__(
        self, 
        model: Optional[TeleChatModel] = None,
        config: Optional[InferenceConfig] = None
    ):
        """
        初始化推理引擎
        
        Args:
            model: TeleChat模型实例
            config: 推理配置
        """
        self.model = model or get_model()
        self.config = config or get_config().inference
        self._executor = ThreadPoolExecutor(max_workers=4)
        
    def generate(self, request: InferenceRequest) -> InferenceResponse:
        """
        同步生成文本
        
        Args:
            request: 推理请求
            
        Returns:
            推理响应
        """
        if not self.model.is_loaded:
            raise ModelException(
                message="模型未加载",
                code=ErrorCodes.MODEL_LOAD_ERROR
            )
        
        start_time = time.time()
        
        try:
            # 构建生成参数
            gen_kwargs = {
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "repetition_penalty": request.repetition_penalty,
                "do_sample": request.do_sample
            }
            
            # 调用模型生成
            generated_text = self.model.generate(
                prompt=request.prompt,
                **gen_kwargs
            )
            
            time_taken = time.time() - start_time
            
            # 估算生成的token数
            tokens_generated = len(self.model.tokenizer.encode(generated_text))
            
            return InferenceResponse(
                text=generated_text,
                request_id=request.request_id,
                tokens_generated=tokens_generated,
                time_taken=time_taken,
                finish_reason="stop"
            )
            
        except Exception as e:
            logger.error(f"推理失败: {str(e)}")
            raise ModelException(
                message=f"推理失败: {str(e)}",
                code=ErrorCodes.MODEL_INFERENCE_ERROR,
                cause=e
            )
    
    def generate_stream(
        self, 
        request: InferenceRequest
    ) -> Generator[str, None, None]:
        """
        流式生成文本
        
        Args:
            request: 推理请求
            
        Yields:
            生成的文本片段
        """
        if not self.model.is_loaded:
            raise ModelException(
                message="模型未加载",
                code=ErrorCodes.MODEL_LOAD_ERROR
            )
        
        try:
            gen_kwargs = {
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "repetition_penalty": request.repetition_penalty,
                "do_sample": request.do_sample
            }
            
            # 使用模型的流式chat接口
            generator = self.model.chat(
                question=request.prompt,
                history=[],
                stream=True,
                **gen_kwargs
            )
            
            for text, _ in generator:
                if text:
                    yield text
                    
        except Exception as e:
            logger.error(f"流式推理失败: {str(e)}")
            raise ModelException(
                message=f"流式推理失败: {str(e)}",
                code=ErrorCodes.MODEL_INFERENCE_ERROR,
                cause=e
            )
    
    async def async_generate(
        self, 
        request: InferenceRequest
    ) -> InferenceResponse:
        """
        异步生成文本
        
        Args:
            request: 推理请求
            
        Returns:
            推理响应
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.generate,
            request
        )
    
    async def async_generate_stream(
        self, 
        request: InferenceRequest
    ) -> AsyncGenerator[str, None]:
        """
        异步流式生成文本
        
        Args:
            request: 推理请求
            
        Yields:
            生成的文本片段
        """
        loop = asyncio.get_event_loop()
        
        # 在线程池中运行同步生成器
        def run_generator():
            return list(self.generate_stream(request))
        
        chunks = await loop.run_in_executor(self._executor, run_generator)
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0)  # 让出控制权
    
    def chat(
        self,
        messages: List[ChatMessage],
        stream: bool = False,
        **kwargs
    ):
        """
        对话接口
        
        Args:
            messages: 对话消息列表
            stream: 是否流式输出
            **kwargs: 其他参数
            
        Returns:
            对话响应
        """
        if not messages:
            raise ModelException(
                message="消息列表不能为空",
                code=ErrorCodes.PARAM_INVALID
            )
        
        # 构建历史和当前问题
        history = []
        current_question = ""
        
        for i, msg in enumerate(messages):
            if i == len(messages) - 1 and msg.role == "user":
                current_question = msg.content
            else:
                history.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        if not current_question:
            current_question = messages[-1].content
        
        # 调用模型chat
        result = self.model.chat(
            question=current_question,
            history=history,
            stream=stream,
            **kwargs
        )
        
        return result
    
    def batch_generate(
        self,
        requests: List[InferenceRequest]
    ) -> List[InferenceResponse]:
        """
        批量生成
        
        Args:
            requests: 请求列表
            
        Returns:
            响应列表
        """
        results = []
        for request in requests:
            try:
                response = self.generate(request)
                results.append(response)
            except Exception as e:
                logger.error(f"批量推理中的请求失败: {str(e)}")
                results.append(InferenceResponse(
                    text="",
                    request_id=request.request_id,
                    finish_reason="error"
                ))
        return results
    
    async def async_batch_generate(
        self,
        requests: List[InferenceRequest]
    ) -> List[InferenceResponse]:
        """
        异步批量生成
        
        Args:
            requests: 请求列表
            
        Returns:
            响应列表
        """
        tasks = [self.async_generate(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取推理统计信息"""
        return {
            "model_loaded": self.model.is_loaded,
            "model_info": self.model.get_model_info() if self.model.is_loaded else None,
            "config": {
                "max_length": self.config.max_length,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p
            }
        }


# 全局推理引擎实例
_global_engine: Optional[InferenceEngine] = None


def get_engine() -> InferenceEngine:
    """获取全局推理引擎"""
    global _global_engine
    if _global_engine is None:
        _global_engine = InferenceEngine()
    return _global_engine
