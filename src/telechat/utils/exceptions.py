"""
TeleChat 异常处理模块
定义统一的异常类和错误码
"""

from typing import Optional, Dict, Any


class ErrorCodes:
    """错误码定义"""
    # 参数错误 100xx
    PARAM_MISSING = "10001"
    PARAM_INVALID = "10002"
    PARAM_OUT_OF_RANGE = "10003"
    
    # 模型错误 101xx
    MODEL_NOT_FOUND = "10101"
    MODEL_LOAD_ERROR = "10102"
    MODEL_INFERENCE_ERROR = "10103"
    
    # 服务错误 102xx
    SERVICE_UNAVAILABLE = "10201"
    SERVICE_TIMEOUT = "10202"
    SERVICE_OVERLOAD = "10203"
    
    # 内存错误 103xx
    MEMORY_ERROR = "10301"
    MEMORY_FULL = "10302"
    MEMORY_RETRIEVE_ERROR = "10303"
    
    # 认证错误 104xx
    AUTH_REQUIRED = "10401"
    AUTH_INVALID = "10402"
    AUTH_EXPIRED = "10403"
    
    # 系统错误 105xx
    SYSTEM_ERROR = "10501"
    GPU_ERROR = "10502"
    IO_ERROR = "10503"


class TeleChatException(Exception):
    """TeleChat基础异常类"""
    
    def __init__(
        self, 
        code: str, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.code = code
        self.message = message
        self.details = details or {}
        self.cause = cause
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details
            }
        }
    
    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class ParamException(TeleChatException):
    """参数异常"""
    
    def __init__(self, message: str, param_name: str = None, **kwargs):
        details = kwargs.get('details', {})
        if param_name:
            details['param_name'] = param_name
        super().__init__(
            code=kwargs.get('code', ErrorCodes.PARAM_INVALID),
            message=message,
            details=details
        )


class ModelException(TeleChatException):
    """模型异常"""
    
    def __init__(self, message: str, model_name: str = None, **kwargs):
        details = kwargs.get('details', {})
        if model_name:
            details['model_name'] = model_name
        super().__init__(
            code=kwargs.get('code', ErrorCodes.MODEL_INFERENCE_ERROR),
            message=message,
            details=details
        )


class ServiceException(TeleChatException):
    """服务异常"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            code=kwargs.get('code', ErrorCodes.SERVICE_UNAVAILABLE),
            message=message,
            details=kwargs.get('details', {})
        )


class MemoryException(TeleChatException):
    """记忆系统异常"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            code=kwargs.get('code', ErrorCodes.MEMORY_ERROR),
            message=message,
            details=kwargs.get('details', {})
        )


class AuthException(TeleChatException):
    """认证异常"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            code=kwargs.get('code', ErrorCodes.AUTH_REQUIRED),
            message=message,
            details=kwargs.get('details', {})
        )


def handle_exception(func):
    """异常处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TeleChatException:
            raise
        except Exception as e:
            raise TeleChatException(
                code=ErrorCodes.SYSTEM_ERROR,
                message=f"系统错误: {str(e)}",
                cause=e
            )
    return wrapper


async def async_handle_exception(func):
    """异步异常处理装饰器"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except TeleChatException:
            raise
        except Exception as e:
            raise TeleChatException(
                code=ErrorCodes.SYSTEM_ERROR,
                message=f"系统错误: {str(e)}",
                cause=e
            )
    return wrapper
