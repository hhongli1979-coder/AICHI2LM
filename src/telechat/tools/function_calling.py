"""
TeleChat 函数调用模块
支持工具使用和函数调用
"""

import json
import inspect
from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass, field
from functools import wraps

from ..utils.logging import get_logger
from ..utils.exceptions import TeleChatException, ErrorCodes

logger = get_logger("telechat.tools")


@dataclass
class FunctionParameter:
    """函数参数定义"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None


@dataclass
class FunctionDefinition:
    """函数定义"""
    name: str
    description: str
    parameters: List[FunctionParameter] = field(default_factory=list)
    function: Optional[Callable] = None
    
    def to_schema(self) -> Dict[str, Any]:
        """转换为JSON Schema格式"""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


class FunctionRegistry:
    """函数注册表"""
    
    def __init__(self):
        self._functions: Dict[str, FunctionDefinition] = {}
    
    def register(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[List[FunctionParameter]] = None
    ):
        """
        注册函数的装饰器
        
        Args:
            name: 函数名称（默认使用函数实际名称）
            description: 函数描述（默认使用docstring）
            parameters: 参数定义（默认从函数签名推断）
        """
        def decorator(func: Callable):
            func_name = name or func.__name__
            func_desc = description or (func.__doc__ or "").strip()
            
            # 如果未提供参数定义，从函数签名推断
            if parameters is None:
                func_params = self._infer_parameters(func)
            else:
                func_params = parameters
            
            # 创建函数定义
            func_def = FunctionDefinition(
                name=func_name,
                description=func_desc,
                parameters=func_params,
                function=func
            )
            
            self._functions[func_name] = func_def
            logger.info(f"注册函数: {func_name}")
            
            return func
        
        return decorator
    
    def _infer_parameters(self, func: Callable) -> List[FunctionParameter]:
        """从函数签名推断参数"""
        params = []
        sig = inspect.signature(func)
        
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }
        
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            
            # 获取类型
            if param.annotation != inspect.Parameter.empty:
                param_type = type_mapping.get(param.annotation, "string")
            else:
                param_type = "string"
            
            # 获取默认值
            has_default = param.default != inspect.Parameter.empty
            
            params.append(FunctionParameter(
                name=name,
                type=param_type,
                description=f"参数 {name}",
                required=not has_default,
                default=param.default if has_default else None
            ))
        
        return params
    
    def get(self, name: str) -> Optional[FunctionDefinition]:
        """获取函数定义"""
        return self._functions.get(name)
    
    def list_functions(self) -> List[str]:
        """列出所有注册的函数"""
        return list(self._functions.keys())
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """获取所有函数的Schema"""
        return [func.to_schema() for func in self._functions.values()]
    
    def call(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        调用函数
        
        Args:
            name: 函数名称
            arguments: 参数字典
            
        Returns:
            函数返回值
        """
        func_def = self._functions.get(name)
        
        if func_def is None:
            raise TeleChatException(
                code=ErrorCodes.PARAM_INVALID,
                message=f"未知函数: {name}"
            )
        
        if func_def.function is None:
            raise TeleChatException(
                code=ErrorCodes.PARAM_INVALID,
                message=f"函数未实现: {name}"
            )
        
        try:
            return func_def.function(**arguments)
        except Exception as e:
            logger.error(f"函数调用失败: {name}, 错误: {str(e)}")
            raise TeleChatException(
                code=ErrorCodes.SYSTEM_ERROR,
                message=f"函数调用失败: {str(e)}"
            )


class FunctionCallingEngine:
    """函数调用引擎"""
    
    def __init__(self):
        self.registry = FunctionRegistry()
        self._setup_builtin_functions()
    
    def _setup_builtin_functions(self):
        """设置内置函数"""
        
        @self.registry.register(
            name="get_current_time",
            description="获取当前时间",
            parameters=[]
        )
        def get_current_time() -> str:
            """获取当前时间"""
            from datetime import datetime
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        @self.registry.register(
            name="calculate",
            description="计算数学表达式",
            parameters=[
                FunctionParameter(
                    name="expression",
                    type="string",
                    description="数学表达式，如 '2 + 3 * 4'"
                )
            ]
        )
        def calculate(expression: str) -> float:
            """计算数学表达式"""
            # 安全的数学计算
            allowed_chars = set("0123456789+-*/().% ")
            if not all(c in allowed_chars for c in expression):
                raise ValueError("表达式包含非法字符")
            return eval(expression)
        
        @self.registry.register(
            name="search_knowledge",
            description="搜索知识库",
            parameters=[
                FunctionParameter(
                    name="query",
                    type="string",
                    description="搜索查询"
                ),
                FunctionParameter(
                    name="limit",
                    type="integer",
                    description="返回结果数量",
                    required=False,
                    default=5
                )
            ]
        )
        def search_knowledge(query: str, limit: int = 5) -> List[str]:
            """搜索知识库"""
            # 这里可以接入实际的知识库
            return [f"知识库搜索结果: {query} (示例)"]
    
    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """注册自定义函数"""
        decorator = self.registry.register(name=name, description=description)
        decorator(func)
    
    def get_available_functions(self) -> List[Dict[str, Any]]:
        """获取可用函数列表"""
        return self.registry.get_schemas()
    
    def execute(self, function_call: Dict[str, Any]) -> Any:
        """
        执行函数调用
        
        Args:
            function_call: 函数调用信息，包含name和arguments
            
        Returns:
            函数执行结果
        """
        name = function_call.get("name")
        arguments = function_call.get("arguments", {})
        
        # 如果arguments是字符串，尝试解析为JSON
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        
        logger.info(f"执行函数: {name}, 参数: {arguments}")
        result = self.registry.call(name, arguments)
        logger.info(f"函数结果: {result}")
        
        return result
    
    def format_function_result(self, name: str, result: Any) -> str:
        """格式化函数结果为文本"""
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)


# 全局函数调用引擎
_global_function_engine: Optional[FunctionCallingEngine] = None


def get_function_engine() -> FunctionCallingEngine:
    """获取全局函数调用引擎"""
    global _global_function_engine
    if _global_function_engine is None:
        _global_function_engine = FunctionCallingEngine()
    return _global_function_engine
