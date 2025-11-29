"""
TeleChat API 主入口
FastAPI应用程序配置和启动
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from .routes import router
from .schemas import ErrorResponse, ErrorDetail
from ..core.model import get_model, load_model
from ..utils.config import get_config, load_config
from ..utils.logging import setup_logging, get_logger
from ..utils.exceptions import TeleChatException

# 设置日志
setup_logging()
logger = get_logger("telechat.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("TeleChat API 服务启动中...")
    
    config = get_config()
    
    # 尝试加载模型
    try:
        if os.path.exists(config.model.path):
            logger.info(f"加载模型: {config.model.path}")
            load_model(config.model.path)
            logger.info("模型加载完成")
        else:
            logger.warning(f"模型路径不存在: {config.model.path}")
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
    
    logger.info("TeleChat API 服务已启动")
    
    yield
    
    # 关闭时
    logger.info("TeleChat API 服务关闭中...")
    model = get_model()
    if model.is_loaded:
        model.unload()
    logger.info("TeleChat API 服务已关闭")


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """创建FastAPI应用"""
    
    # 加载配置
    if config_path:
        load_config(config_path)
    
    config = get_config()
    
    # 创建应用
    app = FastAPI(
        title="TeleChat API",
        description="""
# TeleChat API

星辰语义大模型 API 服务

## 功能特性

- **OpenAI兼容接口**: 支持 `/v1/chat/completions` 标准接口
- **流式输出**: 支持 Server-Sent Events 流式响应
- **多轮对话**: 支持上下文历史管理
- **批量推理**: 支持批量请求处理

## 快速开始

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="telechat-12b",
    messages=[
        {"role": "user", "content": "你好，请介绍一下自己"}
    ]
)
print(response.choices[0].message.content)
```
        """,
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # 添加CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.service.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 注册路由
    app.include_router(router)
    
    # 注册异常处理器
    @app.exception_handler(TeleChatException)
    async def telechat_exception_handler(request: Request, exc: TeleChatException):
        """TeleChat异常处理"""
        logger.error(f"TeleChat Error: [{exc.code}] {exc.message}")
        return JSONResponse(
            status_code=400,
            content=exc.to_dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """通用异常处理"""
        logger.error(f"Unhandled Error: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "10501",
                    "message": f"服务器内部错误: {str(exc)}"
                }
            }
        )
    
    return app


# 创建默认应用实例
app = create_app()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    reload: bool = False,
    workers: int = 1,
    config_path: Optional[str] = None
):
    """运行API服务器"""
    
    if config_path:
        load_config(config_path)
    
    config = get_config()
    
    uvicorn.run(
        "telechat.api.main:app",
        host=host or config.service.host,
        port=port or config.service.port,
        reload=reload or config.service.reload,
        workers=workers or config.service.workers,
        timeout_keep_alive=config.service.timeout
    )


if __name__ == "__main__":
    run_server()
