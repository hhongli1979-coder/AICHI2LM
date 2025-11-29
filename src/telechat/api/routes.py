"""
TeleChat API 路由模块
定义所有API端点
"""

import uuid
import time
import json
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse

from .schemas import (
    ChatRequest, ChatResponse, ChatChoice, ChatStreamResponse, StreamChoice,
    GenerateRequest, GenerateResponse,
    ModelListResponse, ModelInfo,
    HealthResponse, ErrorResponse, ErrorDetail,
    Message, Role, Usage
)
from ..core.model import get_model
from ..core.inference import get_engine, InferenceRequest, ChatMessage
from ..utils.logging import get_logger
from ..utils.exceptions import TeleChatException, ErrorCodes

logger = get_logger("telechat.api")

# 创建路由器
router = APIRouter()


# ==================== 健康检查 ====================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="健康检查",
    description="检查服务状态"
)
async def health_check():
    """健康检查端点"""
    model = get_model()
    return HealthResponse(
        status="healthy" if model.is_loaded else "degraded",
        model=model.config.name,
        model_loaded=model.is_loaded,
        version="2.0.0",
        timestamp=datetime.now().isoformat()
    )


# ==================== 模型信息 ====================

@router.get(
    "/v1/models",
    response_model=ModelListResponse,
    summary="获取模型列表",
    description="列出所有可用模型"
)
async def list_models():
    """获取可用模型列表"""
    model = get_model()
    models = [
        ModelInfo(
            id="telechat-7b",
            created=int(time.time()),
            owned_by="telechat"
        ),
        ModelInfo(
            id="telechat-12b",
            created=int(time.time()),
            owned_by="telechat"
        ),
        ModelInfo(
            id="telechat-12b-v2",
            created=int(time.time()),
            owned_by="telechat"
        )
    ]
    
    # 标记当前加载的模型
    if model.is_loaded:
        for m in models:
            if model.config.name.lower() in m.id:
                m.owned_by = "telechat (active)"
    
    return ModelListResponse(data=models)


@router.get(
    "/v1/models/{model_id}",
    response_model=ModelInfo,
    summary="获取模型信息",
    description="获取指定模型的详细信息"
)
async def get_model_info(model_id: str):
    """获取指定模型信息"""
    return ModelInfo(
        id=model_id,
        created=int(time.time()),
        owned_by="telechat"
    )


# ==================== 对话接口 ====================

@router.post(
    "/v1/chat/completions",
    response_model=ChatResponse,
    summary="对话补全",
    description="OpenAI兼容的对话补全接口"
)
async def chat_completions(request: ChatRequest):
    """
    对话补全接口 (OpenAI兼容)
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    
    try:
        engine = get_engine()
        model = get_model()
        
        if not model.is_loaded:
            raise TeleChatException(
                code=ErrorCodes.MODEL_LOAD_ERROR,
                message="模型未加载"
            )
        
        # 转换消息格式
        messages = [
            ChatMessage(role=m.role, content=m.content)
            for m in request.messages
        ]
        
        # 流式响应
        if request.stream:
            return StreamingResponse(
                _stream_chat_response(
                    engine, messages, request, request_id, created, model.config.name
                ),
                media_type="text/event-stream"
            )
        
        # 非流式响应
        # 构建历史和当前问题
        history = []
        current_question = ""
        
        for i, msg in enumerate(request.messages):
            if i == len(request.messages) - 1 and msg.role in [Role.USER, "user"]:
                current_question = msg.content
            else:
                history.append({
                    "role": "user" if msg.role in [Role.USER, "user"] else "bot",
                    "content": msg.content
                })
        
        if not current_question:
            current_question = request.messages[-1].content
        
        # 调用模型
        answer, _ = model.chat(
            question=current_question,
            history=history,
            stream=False,
            max_length=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty
        )
        
        # 计算token使用量（估算）
        prompt_tokens = sum(len(model.tokenizer.encode(m.content)) for m in request.messages)
        completion_tokens = len(model.tokenizer.encode(answer))
        
        return ChatResponse(
            id=request_id,
            created=created,
            model=model.config.name,
            choices=[
                ChatChoice(
                    index=0,
                    message=Message(role=Role.ASSISTANT, content=answer),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
    except TeleChatException as e:
        logger.error(f"对话请求失败: {e}")
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        logger.error(f"对话请求异常: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_chat_response(
    engine, messages, request, request_id, created, model_name
):
    """流式响应生成器"""
    try:
        model = get_model()
        
        # 构建历史和当前问题
        history = []
        current_question = ""
        
        for i, msg in enumerate(messages):
            if i == len(messages) - 1 and msg.role in ["user", Role.USER]:
                current_question = msg.content
            else:
                history.append({
                    "role": "user" if msg.role in ["user", Role.USER] else "bot",
                    "content": msg.content
                })
        
        if not current_question:
            current_question = messages[-1].content
        
        # 流式生成
        generator = model.chat(
            question=current_question,
            history=history,
            stream=True,
            max_length=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty
        )
        
        for text, _ in generator:
            if text:
                chunk = ChatStreamResponse(
                    id=request_id,
                    created=created,
                    model=model_name,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta={"content": text},
                            finish_reason=None
                        )
                    ]
                )
                yield f"data: {chunk.json()}\n\n"
        
        # 发送结束标记
        final_chunk = ChatStreamResponse(
            id=request_id,
            created=created,
            model=model_name,
            choices=[
                StreamChoice(
                    index=0,
                    delta={},
                    finish_reason="stop"
                )
            ]
        )
        yield f"data: {final_chunk.json()}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"流式响应错误: {str(e)}")
        error_data = {"error": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"


# ==================== 文本生成接口 ====================

@router.post(
    "/v1/completions",
    response_model=GenerateResponse,
    summary="文本生成",
    description="文本补全接口"
)
async def text_completions(request: GenerateRequest):
    """文本生成接口"""
    request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    
    try:
        engine = get_engine()
        model = get_model()
        
        if not model.is_loaded:
            raise TeleChatException(
                code=ErrorCodes.MODEL_LOAD_ERROR,
                message="模型未加载"
            )
        
        # 创建推理请求
        infer_request = InferenceRequest(
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            request_id=request_id
        )
        
        # 执行推理
        response = engine.generate(infer_request)
        
        # 计算token使用量
        prompt_tokens = len(model.tokenizer.encode(request.prompt))
        completion_tokens = response.tokens_generated
        
        return GenerateResponse(
            id=request_id,
            text=response.text,
            finish_reason=response.finish_reason,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
    except TeleChatException as e:
        logger.error(f"生成请求失败: {e}")
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        logger.error(f"生成请求异常: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 兼容旧接口 ====================

@router.post(
    "/telechat/gptDialog/v2",
    summary="对话接口V2（流式）",
    description="兼容旧版流式对话接口"
)
async def legacy_chat_v2(item: dict):
    """兼容旧版流式对话接口"""
    try:
        dialog = item.get("dialog", [])
        if not dialog:
            return {"error": {"code": "10301", "message": "dialog参数缺失"}}
        
        # 转换为新格式
        messages = [
            Message(
                role=Role.USER if d.get("role") == "user" else Role.BOT,
                content=d.get("content", "")
            )
            for d in dialog
        ]
        
        request = ChatRequest(
            messages=messages,
            max_tokens=item.get("max_length", 4096),
            temperature=item.get("temperature", 0.1),
            top_p=item.get("top_p", 0.2),
            top_k=item.get("top_k", 20),
            repetition_penalty=item.get("repetition_penalty", 1.03),
            stream=True
        )
        
        return await chat_completions(request)
        
    except Exception as e:
        logger.error(f"旧版接口错误: {str(e)}")
        return {"error": {"code": "10903", "message": str(e)}}


@router.post(
    "/telechat/gptDialog/v4",
    summary="对话接口V4（非流式）",
    description="兼容旧版非流式对话接口"
)
async def legacy_chat_v4(item: dict):
    """兼容旧版非流式对话接口"""
    try:
        dialog = item.get("dialog", [])
        if not dialog:
            return {"error": {"code": "10301", "message": "dialog参数缺失"}}
        
        # 转换为新格式
        messages = [
            Message(
                role=Role.USER if d.get("role") == "user" else Role.BOT,
                content=d.get("content", "")
            )
            for d in dialog
        ]
        
        request = ChatRequest(
            messages=messages,
            max_tokens=item.get("max_length", 4096),
            temperature=item.get("temperature", 0.1),
            top_p=item.get("top_p", 0.2),
            top_k=item.get("top_k", 20),
            repetition_penalty=item.get("repetition_penalty", 1.03),
            stream=False
        )
        
        response = await chat_completions(request)
        
        # 转换为旧版格式
        return {
            "role": "bot",
            "content": response.choices[0].message.content
        }
        
    except Exception as e:
        logger.error(f"旧版接口错误: {str(e)}")
        return {"error": {"code": "10903", "message": str(e)}}
