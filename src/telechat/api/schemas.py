"""
TeleChat API 数据模型
定义请求和响应的数据结构
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class Role(str, Enum):
    """消息角色"""
    USER = "user"
    BOT = "bot"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """对话消息"""
    role: Role = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容")
    timestamp: Optional[str] = Field(None, description="时间戳")
    
    class Config:
        use_enum_values = True


class ChatRequest(BaseModel):
    """对话请求"""
    messages: List[Message] = Field(..., description="对话消息列表")
    max_tokens: Optional[int] = Field(2048, ge=1, le=8192, description="最大生成token数")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="温度参数")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="核采样参数")
    top_k: Optional[int] = Field(50, ge=1, le=100, description="Top-K采样")
    repetition_penalty: Optional[float] = Field(1.1, ge=1.0, le=2.0, description="重复惩罚")
    stream: Optional[bool] = Field(False, description="是否流式输出")
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    
    class Config:
        schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "你好，请介绍一下自己"}
                ],
                "max_tokens": 2048,
                "temperature": 0.7,
                "stream": False
            }
        }


class ChatChoice(BaseModel):
    """对话选项"""
    index: int = Field(0, description="选项索引")
    message: Message = Field(..., description="回复消息")
    finish_reason: str = Field("stop", description="结束原因")


class Usage(BaseModel):
    """Token使用量"""
    prompt_tokens: int = Field(0, description="输入token数")
    completion_tokens: int = Field(0, description="输出token数")
    total_tokens: int = Field(0, description="总token数")


class ChatResponse(BaseModel):
    """对话响应"""
    id: str = Field(..., description="响应ID")
    object: str = Field("chat.completion", description="对象类型")
    created: int = Field(..., description="创建时间戳")
    model: str = Field(..., description="模型名称")
    choices: List[ChatChoice] = Field(..., description="回复选项")
    usage: Usage = Field(..., description="Token使用量")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "chatcmpl-xxx",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "telechat-12b",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "你好！我是TeleChat..."
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 50,
                    "total_tokens": 60
                }
            }
        }


class StreamChoice(BaseModel):
    """流式响应选项"""
    index: int = Field(0)
    delta: Dict[str, Any] = Field(...)
    finish_reason: Optional[str] = None


class ChatStreamResponse(BaseModel):
    """流式对话响应"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]


class GenerateRequest(BaseModel):
    """文本生成请求"""
    prompt: str = Field(..., description="输入提示")
    max_tokens: Optional[int] = Field(2048, ge=1, le=8192)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(50, ge=1, le=100)
    repetition_penalty: Optional[float] = Field(1.1, ge=1.0, le=2.0)
    stop: Optional[List[str]] = Field(None, description="停止词列表")


class GenerateResponse(BaseModel):
    """文本生成响应"""
    id: str
    text: str
    finish_reason: str = "stop"
    usage: Usage


class ModelInfo(BaseModel):
    """模型信息"""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "telechat"
    
    
class ModelListResponse(BaseModel):
    """模型列表响应"""
    object: str = "list"
    data: List[ModelInfo]


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field("healthy", description="服务状态")
    model: str = Field(..., description="模型名称")
    model_loaded: bool = Field(..., description="模型是否已加载")
    version: str = Field(..., description="API版本")
    timestamp: str = Field(..., description="时间戳")


class ErrorDetail(BaseModel):
    """错误详情"""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """错误响应"""
    error: ErrorDetail
    
    class Config:
        schema_extra = {
            "example": {
                "error": {
                    "code": "10001",
                    "message": "参数错误",
                    "details": {"param": "temperature", "reason": "超出范围"}
                }
            }
        }
