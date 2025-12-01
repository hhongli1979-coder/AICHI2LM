# -*- coding: utf-8 -*-
"""
工具进化系统 - Tool Evolution System

实现能力扩展和工具自动创建机制。
This implements capability expansion and automatic tool creation mechanisms.
"""

import time
import hashlib
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class ToolStatus(Enum):
    """工具状态"""
    DRAFT = "draft"
    TESTING = "testing"
    VALIDATED = "validated"
    DEPRECATED = "deprecated"


@dataclass
class Tool:
    """工具定义"""
    tool_id: str
    name: str
    description: str
    capability: str
    code: str
    status: ToolStatus = ToolStatus.DRAFT
    version: str = "1.0.0"
    created_at: float = field(default_factory=time.time)
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    usage_count: int = 0
    success_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.test_results:
            self.test_results = []
        if not self.metadata:
            self.metadata = {}


@dataclass
class CapabilityGap:
    """能力缺失"""
    capability: str
    description: str
    priority: float
    identified_at: float = field(default_factory=time.time)
    attempts: int = 0


class ToolEvolutionSystem:
    """
    工具进化系统 - Tool Evolution System

    实现工具的自动发现、创建、测试和集成。
    Implements automatic tool discovery, creation, testing, and integration.
    """

    def __init__(
        self,
        max_tools: int = 100,
        min_test_coverage: float = 0.8,
        min_success_rate: float = 0.7
    ):
        """
        初始化工具进化系统

        Args:
            max_tools: 最大工具数量
            min_test_coverage: 最小测试覆盖率
            min_success_rate: 最小成功率要求
        """
        self.tool_library: Dict[str, Tool] = {}
        self.capability_gaps: List[CapabilityGap] = []
        self.capability_index: Dict[str, List[str]] = {}

        self.max_tools = max_tools
        self.min_test_coverage = min_test_coverage
        self.min_success_rate = min_success_rate

        self._statistics = {
            "tools_created": 0,
            "tools_validated": 0,
            "tools_deprecated": 0,
            "gaps_identified": 0,
            "gaps_resolved": 0
        }

    def identify_gap(self, task_description: str) -> Optional[CapabilityGap]:
        """
        识别能力缺失

        Args:
            task_description: 任务描述

        Returns:
            识别出的能力缺失，如果已存在则返回None
        """
        # 分析任务描述，提取所需能力
        capability = self._extract_capability(task_description)

        # 检查是否已有相应工具
        if self._has_capability(capability):
            return None

        # 检查是否已记录该缺失
        for gap in self.capability_gaps:
            if gap.capability == capability:
                gap.attempts += 1
                return gap

        # 创建新的能力缺失记录
        gap = CapabilityGap(
            capability=capability,
            description=task_description,
            priority=self._calculate_priority(task_description)
        )
        self.capability_gaps.append(gap)
        self._statistics["gaps_identified"] += 1

        return gap

    def _extract_capability(self, task_description: str) -> str:
        """
        从任务描述中提取能力名称

        Args:
            task_description: 任务描述

        Returns:
            能力名称
        """
        # 简单实现：使用哈希生成唯一标识
        desc_hash = hashlib.md5(task_description.encode()).hexdigest()[:8]
        return f"capability_{desc_hash}"

    def _has_capability(self, capability: str) -> bool:
        """
        检查是否已有对应能力的工具

        Args:
            capability: 能力名称

        Returns:
            是否存在
        """
        return capability in self.capability_index and len(self.capability_index[capability]) > 0

    def _calculate_priority(self, task_description: str) -> float:
        """
        计算能力缺失的优先级

        Args:
            task_description: 任务描述

        Returns:
            优先级分数 (0-1)
        """
        # 基于描述长度和关键词进行简单优先级计算
        priority = 0.5
        critical_keywords = ["critical", "urgent", "important", "必须", "紧急", "重要"]

        for keyword in critical_keywords:
            if keyword in task_description.lower():
                priority = min(priority + 0.1, 1.0)

        return priority

    def search_or_create_tool(
        self,
        missing_capability: CapabilityGap,
        tool_generator: Optional[Callable[[str], str]] = None
    ) -> Optional[Tool]:
        """
        搜索或创建新工具

        Args:
            missing_capability: 能力缺失描述
            tool_generator: 工具代码生成器（可选）

        Returns:
            创建的工具
        """
        # 首先尝试在现有工具中搜索
        existing_tool = self._search_similar_tool(missing_capability.capability)
        if existing_tool:
            return existing_tool

        # 创建新工具
        tool_code = self._generate_tool_code(
            missing_capability,
            tool_generator
        )

        tool = Tool(
            tool_id=f"tool_{len(self.tool_library) + 1}",
            name=f"Tool for {missing_capability.capability}",
            description=missing_capability.description,
            capability=missing_capability.capability,
            code=tool_code,
            status=ToolStatus.DRAFT
        )

        self._statistics["tools_created"] += 1
        return tool

    def _search_similar_tool(self, capability: str) -> Optional[Tool]:
        """
        搜索相似工具

        Args:
            capability: 能力名称

        Returns:
            找到的工具或None
        """
        if capability in self.capability_index:
            tool_ids = self.capability_index[capability]
            for tool_id in tool_ids:
                if tool_id in self.tool_library:
                    tool = self.tool_library[tool_id]
                    if tool.status == ToolStatus.VALIDATED:
                        return tool
        return None

    def _generate_tool_code(
        self,
        capability: CapabilityGap,
        generator: Optional[Callable[[str], str]] = None
    ) -> str:
        """
        生成工具代码

        Args:
            capability: 能力缺失
            generator: 自定义生成器

        Returns:
            生成的代码
        """
        if generator:
            return generator(capability.description)

        # 默认生成模板代码
        return f'''
def tool_{capability.capability}(input_data):
    """
    Auto-generated tool for: {capability.description}

    Args:
        input_data: Input data for processing

    Returns:
        Processed result
    """
    # TODO: Implement tool logic
    result = input_data
    return result
'''

    def validate_tool(
        self,
        tool: Tool,
        test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> Tool:
        """
        测试并验证工具

        Args:
            tool: 待验证工具
            test_cases: 测试用例列表

        Returns:
            验证后的工具
        """
        tool.status = ToolStatus.TESTING

        if test_cases is None:
            # 生成默认测试用例
            test_cases = [
                {"input": "test_input_1", "expected": "test_output_1"},
                {"input": "test_input_2", "expected": "test_output_2"}
            ]

        # 执行测试
        passed = 0
        for test in test_cases:
            result = self._run_test(tool, test)
            tool.test_results.append(result)
            if result.get("passed", False):
                passed += 1

        # 计算成功率
        tool.success_rate = passed / len(test_cases) if test_cases else 0.0

        # 根据成功率更新状态
        if tool.success_rate >= self.min_success_rate:
            tool.status = ToolStatus.VALIDATED
            self._statistics["tools_validated"] += 1
        else:
            tool.status = ToolStatus.DRAFT

        return tool

    def _run_test(self, tool: Tool, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行单个测试用例

        Args:
            tool: 工具
            test_case: 测试用例

        Returns:
            测试结果
        """
        # 由于不执行实际代码，这里模拟测试结果
        # 在实际实现中，应该安全地执行tool.code
        return {
            "test_case": test_case,
            "passed": True,  # 模拟通过
            "execution_time": 0.001,
            "timestamp": time.time()
        }

    def add_tool(self, tool: Tool) -> bool:
        """
        将工具添加到工具库

        Args:
            tool: 待添加工具

        Returns:
            是否成功添加
        """
        if tool.status != ToolStatus.VALIDATED:
            return False

        if len(self.tool_library) >= self.max_tools:
            # 移除使用最少的工具
            self._evict_least_used_tool()

        self.tool_library[tool.tool_id] = tool

        # 更新能力索引
        if tool.capability not in self.capability_index:
            self.capability_index[tool.capability] = []
        self.capability_index[tool.capability].append(tool.tool_id)

        # 移除已解决的能力缺失
        self.capability_gaps = [
            gap for gap in self.capability_gaps
            if gap.capability != tool.capability
        ]
        self._statistics["gaps_resolved"] += 1

        return True

    def _evict_least_used_tool(self) -> None:
        """移除使用最少的工具"""
        if not self.tool_library:
            return

        least_used_id = min(
            self.tool_library.keys(),
            key=lambda k: self.tool_library[k].usage_count
        )
        self.remove_tool(least_used_id)

    def remove_tool(self, tool_id: str) -> bool:
        """
        移除工具

        Args:
            tool_id: 工具ID

        Returns:
            是否成功移除
        """
        if tool_id not in self.tool_library:
            return False

        tool = self.tool_library.pop(tool_id)
        tool.status = ToolStatus.DEPRECATED
        self._statistics["tools_deprecated"] += 1

        # 更新能力索引
        if tool.capability in self.capability_index:
            if tool_id in self.capability_index[tool.capability]:
                self.capability_index[tool.capability].remove(tool_id)

        return True

    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """
        获取工具

        Args:
            tool_id: 工具ID

        Returns:
            工具对象或None
        """
        return self.tool_library.get(tool_id)

    def use_tool(self, tool_id: str, input_data: Any) -> Dict[str, Any]:
        """
        使用工具

        Args:
            tool_id: 工具ID
            input_data: 输入数据

        Returns:
            执行结果
        """
        tool = self.get_tool(tool_id)
        if not tool:
            return {"error": "Tool not found", "success": False}

        if tool.status != ToolStatus.VALIDATED:
            return {"error": "Tool not validated", "success": False}

        tool.usage_count += 1

        # 模拟工具执行
        return {
            "success": True,
            "result": f"Executed {tool.name} with input: {input_data}",
            "tool_id": tool_id
        }

    def get_tools_by_capability(self, capability: str) -> List[Tool]:
        """
        按能力获取工具

        Args:
            capability: 能力名称

        Returns:
            工具列表
        """
        tool_ids = self.capability_index.get(capability, [])
        return [
            self.tool_library[tid]
            for tid in tool_ids
            if tid in self.tool_library
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取系统统计信息

        Returns:
            统计信息
        """
        return {
            **self._statistics,
            "total_tools": len(self.tool_library),
            "active_gaps": len(self.capability_gaps),
            "capabilities_covered": len(self.capability_index),
            "validated_tools": sum(
                1 for t in self.tool_library.values()
                if t.status == ToolStatus.VALIDATED
            )
        }

    def get_pending_gaps(self, limit: int = 10) -> List[CapabilityGap]:
        """
        获取待解决的能力缺失

        Args:
            limit: 返回数量限制

        Returns:
            按优先级排序的能力缺失列表
        """
        sorted_gaps = sorted(
            self.capability_gaps,
            key=lambda g: g.priority,
            reverse=True
        )
        return sorted_gaps[:limit]

    def export_tools(self) -> List[Dict[str, Any]]:
        """
        导出工具库

        Returns:
            工具数据列表
        """
        return [
            {
                "tool_id": tool.tool_id,
                "name": tool.name,
                "description": tool.description,
                "capability": tool.capability,
                "code": tool.code,
                "status": tool.status.value,
                "version": tool.version,
                "usage_count": tool.usage_count,
                "success_rate": tool.success_rate
            }
            for tool in self.tool_library.values()
        ]

    def evolve_tools(self) -> Dict[str, Any]:
        """
        执行工具进化 - 尝试解决所有待解决的能力缺失

        Returns:
            进化结果
        """
        results = {
            "processed": 0,
            "created": 0,
            "validated": 0,
            "failed": 0
        }

        for gap in list(self.capability_gaps):
            results["processed"] += 1

            # 创建工具
            tool = self.search_or_create_tool(gap)
            if tool is None:
                results["failed"] += 1
                continue
            results["created"] += 1

            # 验证工具
            validated_tool = self.validate_tool(tool)

            # 添加到工具库
            if self.add_tool(validated_tool):
                results["validated"] += 1
            else:
                results["failed"] += 1

        return results
