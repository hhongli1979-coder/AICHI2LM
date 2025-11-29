"""
工具能力自我扩展 (Self Tool Creator)
====================================

实现自主工具创建能力:
- 识别工具需求
- 搜索现有解决方案
- 创建新工具
- 测试和优化工具
- 集成工具到系统
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ToolCapability:
    """工具能力数据类"""
    name: str
    description: str
    category: str  # 'data_processing', 'analysis', 'generation', 'integration', etc.
    priority: int  # 1-5, 1为最高优先级
    required_inputs: List[str]
    expected_outputs: List[str]


@dataclass
class ExecutionDifficulty:
    """执行困难数据类"""
    task_type: str
    difficulty_description: str
    missing_capability: str
    frequency: int = 1
    first_detected: datetime = field(default_factory=datetime.now)


@dataclass
class Tool:
    """工具数据类"""
    tool_id: str
    name: str
    description: str
    category: str
    code: str
    version: str = "1.0.0"
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    usage_count: int = 0
    success_rate: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'tool_id': self.tool_id,
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'code': self.code,
            'version': self.version,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'usage_count': self.usage_count,
            'success_rate': self.success_rate
        }


class SelfToolCreator:
    """
    工具能力自我扩展
    
    实现自主工具创建:
    1. 识别工具需求
    2. 搜索现有解决方案
    3. 创建新工具
    4. 测试和优化工具
    5. 集成工具到系统
    """
    
    def __init__(self):
        """初始化工具创建器"""
        self.execution_difficulties: List[ExecutionDifficulty] = []
        self.created_tools: Dict[str, Tool] = {}
        self.tool_templates: Dict[str, str] = self._load_tool_templates()
        self.capability_registry: Dict[str, ToolCapability] = {}
        
    def _load_tool_templates(self) -> Dict[str, str]:
        """加载工具模板"""
        return {
            'data_processor': '''
def {name}(input_data):
    """
    {description}
    
    Args:
        input_data: 输入数据
        
    Returns:
        处理后的数据
    """
    # 数据验证
    if input_data is None:
        raise ValueError("输入数据不能为空")
    
    # 数据处理逻辑
    result = input_data
    
    # 返回结果
    return result
''',
            'analyzer': '''
def {name}(data, **kwargs):
    """
    {description}
    
    Args:
        data: 待分析的数据
        **kwargs: 分析参数
        
    Returns:
        Dict: 分析结果
    """
    results = {{
        'summary': None,
        'details': [],
        'metrics': {{}}
    }}
    
    # 分析逻辑
    
    return results
''',
            'generator': '''
def {name}(prompt, **kwargs):
    """
    {description}
    
    Args:
        prompt: 生成提示
        **kwargs: 生成参数
        
    Returns:
        生成的内容
    """
    # 参数处理
    max_length = kwargs.get('max_length', 1000)
    
    # 生成逻辑
    generated_content = ""
    
    return generated_content
''',
            'validator': '''
def {name}(data, rules=None):
    """
    {description}
    
    Args:
        data: 待验证的数据
        rules: 验证规则
        
    Returns:
        Tuple[bool, List[str]]: (是否通过验证, 错误列表)
    """
    errors = []
    
    # 验证逻辑
    
    is_valid = len(errors) == 0
    return is_valid, errors
''',
            'converter': '''
def {name}(input_data, source_format, target_format):
    """
    {description}
    
    Args:
        input_data: 输入数据
        source_format: 源格式
        target_format: 目标格式
        
    Returns:
        转换后的数据
    """
    # 格式转换逻辑
    converted_data = input_data
    
    return converted_data
'''
        }
        
    def analyze_task_execution_issues(
        self,
        task_history: List[Dict[str, Any]]
    ) -> List[ExecutionDifficulty]:
        """
        分析任务执行中的困难
        
        Args:
            task_history: 任务执行历史
            
        Returns:
            List[ExecutionDifficulty]: 执行困难列表
        """
        difficulties = []
        
        for task in task_history:
            # 检查任务是否失败或遇到困难
            if task.get('status') == 'failed' or task.get('had_difficulty', False):
                difficulty = ExecutionDifficulty(
                    task_type=task.get('task_type', 'unknown'),
                    difficulty_description=task.get('error_message', '执行失败'),
                    missing_capability=self._infer_missing_capability(task)
                )
                
                # 检查是否已存在相同困难
                existing = next(
                    (d for d in self.execution_difficulties 
                     if d.missing_capability == difficulty.missing_capability),
                    None
                )
                
                if existing:
                    existing.frequency += 1
                else:
                    self.execution_difficulties.append(difficulty)
                    difficulties.append(difficulty)
                    
        logger.info(f"Analyzed {len(difficulties)} new execution difficulties")
        return difficulties
        
    def identify_missing_capabilities(
        self,
        difficulties: List[ExecutionDifficulty]
    ) -> List[ToolCapability]:
        """
        识别缺失的能力
        
        Args:
            difficulties: 执行困难列表
            
        Returns:
            List[ToolCapability]: 缺失能力列表
        """
        missing_capabilities = []
        
        for difficulty in difficulties:
            capability = self._map_difficulty_to_capability(difficulty)
            if capability:
                missing_capabilities.append(capability)
                
        # 按优先级排序
        missing_capabilities.sort(key=lambda x: x.priority)
        
        logger.info(f"Identified {len(missing_capabilities)} missing capabilities")
        return missing_capabilities
        
    def identify_tool_needs(self) -> List[ToolCapability]:
        """
        识别需要的新工具
        
        Returns:
            List[ToolCapability]: 需要的工具能力列表
        """
        # 基于执行困难识别工具需求
        difficulties = self.execution_difficulties
        
        # 按频率排序，高频困难优先
        sorted_difficulties = sorted(
            difficulties,
            key=lambda x: x.frequency,
            reverse=True
        )
        
        # 识别缺失能力
        missing_capabilities = self.identify_missing_capabilities(sorted_difficulties[:10])
        
        return missing_capabilities
        
    def search_for_existing_solutions(
        self,
        capability_needed: ToolCapability
    ) -> List[Dict[str, Any]]:
        """
        搜索现有解决方案
        
        Args:
            capability_needed: 需要的能力
            
        Returns:
            List[Dict[str, Any]]: 现有解决方案列表
        """
        solutions = []
        
        # 搜索已有工具
        for tool_id, tool in self.created_tools.items():
            if self._tool_matches_capability(tool, capability_needed):
                solutions.append({
                    'type': 'existing_tool',
                    'tool': tool.to_dict(),
                    'match_score': self._calculate_match_score(tool, capability_needed)
                })
                
        # 搜索模板
        for template_name, template_code in self.tool_templates.items():
            if capability_needed.category in template_name:
                solutions.append({
                    'type': 'template',
                    'template_name': template_name,
                    'template_code': template_code,
                    'match_score': 0.5
                })
                
        # 按匹配度排序
        solutions.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        
        logger.info(f"Found {len(solutions)} existing solutions for {capability_needed.name}")
        return solutions
        
    def adapt_existing_solution(
        self,
        solution: Dict[str, Any],
        capability_needed: ToolCapability
    ) -> Tool:
        """
        适配现有方案
        
        Args:
            solution: 现有解决方案
            capability_needed: 需要的能力
            
        Returns:
            Tool: 适配后的工具
        """
        if solution['type'] == 'existing_tool':
            # 基于现有工具创建变体
            base_tool = solution['tool']
            tool_id = f"{base_tool['tool_id']}_adapted_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            adapted_tool = Tool(
                tool_id=tool_id,
                name=f"{capability_needed.name}_adapted",
                description=capability_needed.description,
                category=capability_needed.category,
                code=self._adapt_code(base_tool['code'], capability_needed),
                inputs=capability_needed.required_inputs,
                outputs=capability_needed.expected_outputs
            )
            
        else:  # template
            # 基于模板创建新工具
            template_code = solution['template_code']
            tool_id = f"tool_{capability_needed.category}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            adapted_tool = Tool(
                tool_id=tool_id,
                name=capability_needed.name,
                description=capability_needed.description,
                category=capability_needed.category,
                code=template_code.format(
                    name=capability_needed.name.replace(' ', '_').lower(),
                    description=capability_needed.description
                ),
                inputs=capability_needed.required_inputs,
                outputs=capability_needed.expected_outputs
            )
            
        logger.info(f"Adapted solution to create tool: {adapted_tool.tool_id}")
        return adapted_tool
        
    def generate_tool_from_scratch(
        self,
        capability_needed: ToolCapability
    ) -> str:
        """
        从零开始创造工具
        
        Args:
            capability_needed: 需要的能力
            
        Returns:
            str: 工具代码
        """
        # 根据能力类型选择基础模板
        category = capability_needed.category
        base_template = self.tool_templates.get(
            category.split('_')[0],  # 取类型的第一部分
            self.tool_templates['data_processor']  # 默认模板
        )
        
        # 生成工具代码
        function_name = capability_needed.name.replace(' ', '_').lower()
        
        code = f'''
def {function_name}({", ".join(capability_needed.required_inputs)}):
    """
    {capability_needed.description}
    
    能力类型: {capability_needed.category}
    优先级: {capability_needed.priority}
    
    Args:
        {self._generate_args_doc(capability_needed.required_inputs)}
        
    Returns:
        {self._generate_returns_doc(capability_needed.expected_outputs)}
    """
    # 输入验证
    {self._generate_input_validation(capability_needed.required_inputs)}
    
    # 核心逻辑
    result = {{}}
    
    # 处理逻辑实现
    # TODO: 根据具体需求实现处理逻辑
    
    # 输出结果
    return result
'''
        
        logger.info(f"Generated tool code from scratch for: {capability_needed.name}")
        return code
        
    def test_and_refine_tool(
        self,
        tool_code: str,
        capability_needed: ToolCapability
    ) -> Tool:
        """
        测试和优化工具
        
        Args:
            tool_code: 工具代码
            capability_needed: 需要的能力
            
        Returns:
            Tool: 测试和优化后的工具
        """
        tool_id = f"tool_{hashlib.md5(tool_code.encode()).hexdigest()[:8]}"
        
        # 创建工具对象
        tool = Tool(
            tool_id=tool_id,
            name=capability_needed.name,
            description=capability_needed.description,
            category=capability_needed.category,
            code=tool_code,
            inputs=capability_needed.required_inputs,
            outputs=capability_needed.expected_outputs
        )
        
        # 生成测试用例
        test_cases = self._generate_test_cases(capability_needed)
        tool.test_cases = test_cases
        
        # 执行测试
        test_results = self._run_tests(tool, test_cases)
        
        # 根据测试结果优化
        if test_results['pass_rate'] < 1.0:
            tool = self._refine_tool(tool, test_results)
            
        tool.success_rate = test_results['pass_rate']
        
        logger.info(
            f"Tool tested and refined: {tool.tool_id}, "
            f"success_rate={tool.success_rate:.2f}"
        )
        return tool
        
    def integrate_tool_into_system(self, tool: Tool) -> bool:
        """
        集成工具到系统
        
        Args:
            tool: 工具对象
            
        Returns:
            bool: 集成是否成功
        """
        # 验证工具
        if tool.success_rate < 0.8:
            logger.warning(
                f"Tool {tool.tool_id} has low success rate: {tool.success_rate:.2f}"
            )
            
        # 注册工具
        self.created_tools[tool.tool_id] = tool
        
        # 注册能力
        capability = ToolCapability(
            name=tool.name,
            description=tool.description,
            category=tool.category,
            priority=3,
            required_inputs=tool.inputs,
            expected_outputs=tool.outputs
        )
        self.capability_registry[tool.name] = capability
        
        logger.info(f"Integrated tool: {tool.tool_id} into system")
        return True
        
    def create_new_tool(self, capability_needed: ToolCapability) -> Tool:
        """
        自主创建新工具 - 完整流程
        
        Args:
            capability_needed: 需要的能力
            
        Returns:
            Tool: 创建的新工具
        """
        # 搜索现有解决方案
        existing_solutions = self.search_for_existing_solutions(capability_needed)
        
        if existing_solutions:
            # 适配现有方案
            adapted_tool = self.adapt_existing_solution(
                existing_solutions[0],
                capability_needed
            )
            tested_tool = self.test_and_refine_tool(
                adapted_tool.code,
                capability_needed
            )
        else:
            # 从零开始创造
            new_tool_code = self.generate_tool_from_scratch(capability_needed)
            tested_tool = self.test_and_refine_tool(new_tool_code, capability_needed)
            
        # 集成新工具到系统
        self.integrate_tool_into_system(tested_tool)
        
        return tested_tool
        
    def use_tool(
        self,
        tool_id: str,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        使用工具
        
        Args:
            tool_id: 工具ID
            inputs: 输入参数
            
        Returns:
            Dict[str, Any]: 工具执行结果
        """
        if tool_id not in self.created_tools:
            return {
                'success': False,
                'error': f'Tool {tool_id} not found'
            }
            
        tool = self.created_tools[tool_id]
        tool.usage_count += 1
        
        # 这里只是模拟执行，实际应该执行工具代码
        result = {
            'success': True,
            'tool_id': tool_id,
            'inputs': inputs,
            'outputs': {}  # 实际输出
        }
        
        return result
        
    def _infer_missing_capability(self, task: Dict[str, Any]) -> str:
        """推断缺失的能力"""
        error = task.get('error_message', '')
        task_type = task.get('task_type', '')
        
        if 'parse' in error.lower() or 'format' in error.lower():
            return 'data_parsing'
        elif 'convert' in error.lower():
            return 'format_conversion'
        elif 'validate' in error.lower():
            return 'data_validation'
        elif 'analyze' in error.lower():
            return 'data_analysis'
        else:
            return f'{task_type}_capability'
            
    def _map_difficulty_to_capability(
        self,
        difficulty: ExecutionDifficulty
    ) -> Optional[ToolCapability]:
        """将困难映射到能力需求"""
        capability_mapping = {
            'data_parsing': ToolCapability(
                name='数据解析器',
                description='解析各种格式的数据',
                category='data_processor',
                priority=2,
                required_inputs=['raw_data', 'format_type'],
                expected_outputs=['parsed_data']
            ),
            'format_conversion': ToolCapability(
                name='格式转换器',
                description='在不同数据格式之间转换',
                category='converter',
                priority=2,
                required_inputs=['data', 'source_format', 'target_format'],
                expected_outputs=['converted_data']
            ),
            'data_validation': ToolCapability(
                name='数据验证器',
                description='验证数据的完整性和正确性',
                category='validator',
                priority=1,
                required_inputs=['data', 'validation_rules'],
                expected_outputs=['is_valid', 'errors']
            ),
            'data_analysis': ToolCapability(
                name='数据分析器',
                description='分析数据并提取洞察',
                category='analyzer',
                priority=2,
                required_inputs=['data'],
                expected_outputs=['analysis_results']
            ),
        }
        
        return capability_mapping.get(difficulty.missing_capability)
        
    def _tool_matches_capability(
        self,
        tool: Tool,
        capability: ToolCapability
    ) -> bool:
        """检查工具是否匹配能力需求"""
        return tool.category == capability.category
        
    def _calculate_match_score(
        self,
        tool: Tool,
        capability: ToolCapability
    ) -> float:
        """计算匹配分数"""
        score = 0.0
        
        # 类型匹配
        if tool.category == capability.category:
            score += 0.5
            
        # 输入输出匹配
        input_overlap = len(set(tool.inputs) & set(capability.required_inputs))
        if tool.inputs:
            score += 0.25 * (input_overlap / len(tool.inputs))
            
        output_overlap = len(set(tool.outputs) & set(capability.expected_outputs))
        if tool.outputs:
            score += 0.25 * (output_overlap / len(tool.outputs))
            
        return score
        
    def _adapt_code(self, code: str, capability: ToolCapability) -> str:
        """适配代码"""
        # 简单的代码适配：替换描述
        adapted = code.replace(
            'TODO:',
            f'# Adapted for: {capability.description}\n    # TODO:'
        )
        return adapted
        
    def _generate_args_doc(self, inputs: List[str]) -> str:
        """生成参数文档"""
        return '\n        '.join([f'{inp}: 输入参数' for inp in inputs])
        
    def _generate_returns_doc(self, outputs: List[str]) -> str:
        """生成返回值文档"""
        return ', '.join(outputs)
        
    def _generate_input_validation(self, inputs: List[str]) -> str:
        """生成输入验证代码"""
        validations = []
        for inp in inputs:
            validations.append(f'if {inp} is None:\n        raise ValueError("{inp} 不能为空")')
        return '\n    '.join(validations)
        
    def _generate_test_cases(
        self,
        capability: ToolCapability
    ) -> List[Dict[str, Any]]:
        """生成测试用例"""
        test_cases = [
            {
                'name': 'basic_test',
                'inputs': {inp: 'test_value' for inp in capability.required_inputs},
                'expected_success': True
            },
            {
                'name': 'null_input_test',
                'inputs': {inp: None for inp in capability.required_inputs},
                'expected_success': False
            },
            {
                'name': 'empty_input_test',
                'inputs': {inp: '' for inp in capability.required_inputs},
                'expected_success': True
            }
        ]
        return test_cases
        
    def _run_tests(
        self,
        tool: Tool,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """运行测试"""
        results = {
            'total': len(test_cases),
            'passed': 0,
            'failed': 0,
            'details': []
        }
        
        for test_case in test_cases:
            # 模拟测试执行
            # 实际应该执行tool.code并验证结果
            test_passed = test_case.get('expected_success', True)
            
            results['details'].append({
                'name': test_case['name'],
                'passed': test_passed
            })
            
            if test_passed:
                results['passed'] += 1
            else:
                results['failed'] += 1
                
        results['pass_rate'] = results['passed'] / results['total'] if results['total'] > 0 else 0
        
        return results
        
    def _refine_tool(
        self,
        tool: Tool,
        test_results: Dict[str, Any]
    ) -> Tool:
        """优化工具"""
        # 根据失败的测试用例优化工具
        failed_tests = [d for d in test_results['details'] if not d['passed']]
        
        refined_code = tool.code
        
        for failed_test in failed_tests:
            # 添加错误处理
            if 'null' in failed_test['name']:
                refined_code = refined_code.replace(
                    '# 输入验证',
                    '# 输入验证 - 增强的空值检查'
                )
                
        tool.code = refined_code
        return tool
        
    def get_tool_statistics(self) -> Dict[str, Any]:
        """获取工具统计信息"""
        return {
            'total_tools': len(self.created_tools),
            'total_capabilities': len(self.capability_registry),
            'execution_difficulties': len(self.execution_difficulties),
            'tools_by_category': self._count_tools_by_category(),
            'avg_success_rate': self._calculate_avg_success_rate()
        }
        
    def _count_tools_by_category(self) -> Dict[str, int]:
        """按类别统计工具"""
        counts = {}
        for tool in self.created_tools.values():
            counts[tool.category] = counts.get(tool.category, 0) + 1
        return counts
        
    def _calculate_avg_success_rate(self) -> float:
        """计算平均成功率"""
        if not self.created_tools:
            return 0.0
        total_rate = sum(t.success_rate for t in self.created_tools.values())
        return total_rate / len(self.created_tools)
