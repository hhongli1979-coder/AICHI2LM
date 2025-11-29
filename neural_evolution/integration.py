# -*- coding: utf-8 -*-
"""
神经进化系统集成 - Neural Evolution System Integration

集成所有进化组件，提供统一的进化系统接口。
Integrates all evolution components into a unified system interface.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .darwin_godel_machine import DarwinGodelMachine, Agent
from .evolutionary_memory import EvolutionaryMemory, Experience
from .tool_evolution import ToolEvolutionSystem
from .multi_round_thinking import MultiRoundThinking
from .self_reward import SelfRewardingSystem
from .evolution_monitor import EvolutionMonitor, MetricType
from .evolution_laws import EvolutionaryLaws, Modification, ModificationType
from .multimodal_brain import UnifiedMultimodalBrain


@dataclass
class EvolutionConfig:
    """进化系统配置"""
    population_size: int = 10
    mutation_rate: float = 0.1
    thinking_rounds: int = 3
    safety_threshold: float = 0.8
    performance_threshold: float = 0.0
    learning_rate: float = 0.1


class NeuralEvolutionSystem:
    """
    神经进化系统 - Neural Evolution System

    整合达尔文哥德尔机、记忆进化、工具进化、多轮思考、
    自我奖励、进化监控和安全定律等组件，实现完整的自进化能力。

    This system integrates Darwin-Gödel Machine, evolutionary memory,
    tool evolution, multi-round thinking, self-reward, evolution monitoring,
    and safety laws for complete self-evolution capabilities.
    """

    def __init__(self, config: Optional[EvolutionConfig] = None):
        """
        初始化神经进化系统

        Args:
            config: 进化配置
        """
        self.config = config or EvolutionConfig()

        # 初始化各个子系统
        self.darwin_machine = DarwinGodelMachine(
            population_size=self.config.population_size,
            mutation_rate=self.config.mutation_rate
        )

        self.memory = EvolutionaryMemory()

        self.tool_system = ToolEvolutionSystem()

        self.thinking = MultiRoundThinking(
            thinking_rounds=self.config.thinking_rounds
        )

        self.reward_system = SelfRewardingSystem(
            learning_rate=self.config.learning_rate
        )

        self.monitor = EvolutionMonitor()

        self.laws = EvolutionaryLaws(
            safety_threshold=self.config.safety_threshold,
            performance_threshold=self.config.performance_threshold
        )

        self.multimodal = UnifiedMultimodalBrain()

        # 系统状态
        self._initialized: bool = False
        self._evolution_count: int = 0
        self._last_evolution: float = 0.0

    def initialize(
        self,
        base_capabilities: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        初始化系统

        Args:
            base_capabilities: 基础能力配置
        """
        # 初始化智能体种群
        self.darwin_machine.initialize_population(base_capabilities)

        self._initialized = True

    def evolve(self, problem: Optional[str] = None) -> Dict[str, Any]:
        """
        执行一次完整的进化循环

        Args:
            problem: 可选的问题用于训练

        Returns:
            进化结果
        """
        if not self._initialized:
            self.initialize()

        results = {
            "timestamp": time.time(),
            "evolution_count": self._evolution_count + 1,
            "components": {}
        }

        # 1. 达尔文哥德尔机进化
        darwin_result = self.darwin_machine.evolve()
        results["components"]["darwin"] = {
            "performance": darwin_result.performance,
            "improvement": darwin_result.improvement,
            "generation": darwin_result.generation
        }

        # 2. 多轮思考训练（如果提供了问题）
        if problem:
            best_solution, solutions = self.thinking.train_self(problem)
            results["components"]["thinking"] = {
                "rounds": len(solutions),
                "final_score": best_solution.score,
                "quality": best_solution.quality.value
            }

            # 记录经验到记忆系统
            experience = Experience(
                task_type="problem_solving",
                input_data=problem,
                output_data=best_solution.content,
                success=best_solution.score >= 0.7,
                score=best_solution.score
            )
            self.memory.evolve_memory(experience)

            # 自我评估和奖励
            score, reward = self.reward_system.evaluate_own_performance(
                problem, best_solution.content
            )
            results["components"]["reward"] = {
                "score": score,
                "reward_type": reward.reward_type.value,
                "reward_value": reward.value
            }

        # 3. 工具进化
        tool_results = self.tool_system.evolve_tools()
        results["components"]["tools"] = tool_results

        # 4. 进化监控
        metrics = self.monitor.track_evolution()
        results["components"]["metrics"] = metrics

        # 5. 安全检查
        modification = Modification(
            modification_id=f"evo_{self._evolution_count}",
            modification_type=ModificationType.PARAMETER_CHANGE,
            description="常规进化更新",
            changes={"performance": darwin_result.performance},
            expected_impact=darwin_result.improvement
        )
        safety_check = self.laws.law1_endure(modification)
        results["components"]["safety"] = {
            "passed": safety_check.passed,
            "safety_level": safety_check.safety_level.value,
            "score": safety_check.score
        }

        # 更新系统状态
        self._evolution_count += 1
        self._last_evolution = time.time()

        # 记录监控指标
        self.monitor.record_metric(
            MetricType.INTELLIGENCE,
            darwin_result.performance
        )
        self.monitor.record_metric(
            MetricType.LEARNING_SPEED,
            darwin_result.improvement if darwin_result.improvement > 0 else 0
        )

        return results

    def think_and_solve(
        self,
        problem: str,
        max_rounds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        思考并解决问题

        Args:
            problem: 问题描述
            max_rounds: 最大思考轮数

        Returns:
            解决方案和过程
        """
        if max_rounds:
            original_rounds = self.thinking.thinking_rounds
            self.thinking.thinking_rounds = max_rounds

        best, solutions = self.thinking.train_self(problem)

        if max_rounds:
            self.thinking.thinking_rounds = original_rounds

        # 评估解决方案
        score, reward = self.reward_system.evaluate_own_performance(
            problem, best.content
        )

        # 获取相关上下文
        context = self.memory.get_relevant_context("problem_solving")

        return {
            "solution": best.content,
            "score": best.score,
            "quality": best.quality.value,
            "rounds": len(solutions),
            "reasoning": best.reasoning_steps,
            "evaluation": {
                "score": score,
                "reward": reward.reward_type.value
            },
            "context": context
        }

    def learn_from_experience(
        self,
        task: str,
        result: str,
        success: bool,
        score: float
    ) -> Dict[str, Any]:
        """
        从经验中学习

        Args:
            task: 任务描述
            result: 执行结果
            success: 是否成功
            score: 得分

        Returns:
            学习结果
        """
        # 创建经验记录
        experience = Experience(
            task_type=self._classify_task(task),
            input_data=task,
            output_data=result,
            success=success,
            score=score
        )

        # 进化记忆
        insight = self.memory.evolve_memory(experience)

        # 自我评估
        eval_score, reward = self.reward_system.evaluate_own_performance(
            task, result
        )

        # 强化学习
        update_info = self.reward_system.reinforce_learning(reward)

        return {
            "experience_added": True,
            "insight_created": insight is not None,
            "insight": insight.content if insight else None,
            "evaluation": eval_score,
            "reward": reward.value,
            "learning_update": update_info
        }

    def _classify_task(self, task: str) -> str:
        """
        分类任务类型

        Args:
            task: 任务描述

        Returns:
            任务类型
        """
        keywords = {
            "代码": "coding",
            "编程": "coding",
            "写": "writing",
            "翻译": "translation",
            "数学": "math",
            "分析": "analysis"
        }

        for keyword, task_type in keywords.items():
            if keyword in task:
                return task_type

        return "general"

    def request_capability(self, description: str) -> Dict[str, Any]:
        """
        请求新能力

        Args:
            description: 能力描述

        Returns:
            能力请求结果
        """
        # 识别能力缺失
        gap = self.tool_system.identify_gap(description)

        if gap is None:
            return {
                "success": True,
                "message": "已存在对应能力",
                "existing": True
            }

        # 尝试创建工具
        tool = self.tool_system.search_or_create_tool(gap)
        if tool is None:
            return {
                "success": False,
                "message": "无法创建对应工具"
            }

        # 验证工具
        validated = self.tool_system.validate_tool(tool)

        # 添加到工具库
        added = self.tool_system.add_tool(validated)

        return {
            "success": added,
            "tool_id": tool.tool_id if added else None,
            "capability": gap.capability,
            "status": validated.status.value
        }

    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态

        Returns:
            系统状态
        """
        return {
            "initialized": self._initialized,
            "evolution_count": self._evolution_count,
            "last_evolution": self._last_evolution,
            "components": {
                "darwin_machine": self.darwin_machine.get_statistics(),
                "memory": self.memory.get_statistics(),
                "tools": self.tool_system.get_statistics(),
                "thinking": self.thinking.get_statistics(),
                "reward": self.reward_system.get_statistics(),
                "monitor": self.monitor.get_statistics(),
                "laws": self.laws.get_statistics(),
                "multimodal": self.multimodal.get_statistics()
            }
        }

    def get_evolution_metrics(self) -> Dict[str, Any]:
        """
        获取进化指标

        Returns:
            进化指标
        """
        return self.monitor.track_evolution()

    def process_multimodal_input(
        self,
        text: Optional[str] = None,
        voice: Optional[Any] = None,
        image: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        处理多模态输入

        Args:
            text: 文本
            voice: 语音
            image: 图像

        Returns:
            处理结果
        """
        representation = self.multimodal.process_multimodal(
            text=text,
            voice=voice,
            image=image
        )

        return {
            "modalities": [m.value for m in representation.modalities],
            "alignment_score": representation.alignment_score,
            "features": {k: str(v)[:100] for k, v in representation.features.items()}
        }

    def safe_evolve(
        self,
        modification_description: str,
        changes: Dict[str, Any],
        expected_impact: float = 0.0
    ) -> Dict[str, Any]:
        """
        安全进化 - 在三定律约束下进行进化

        Args:
            modification_description: 修改描述
            changes: 修改内容
            expected_impact: 预期影响

        Returns:
            进化结果
        """
        modification = Modification(
            modification_id=f"safe_evo_{self._evolution_count}",
            modification_type=ModificationType.PARAMETER_CHANGE,
            description=modification_description,
            changes=changes,
            expected_impact=expected_impact
        )

        result = self.laws.apply_evolution(modification)

        if result["success"]:
            # 执行实际进化
            evolution_result = self.evolve()
            result["evolution"] = evolution_result

        return result

    def export_state(self) -> Dict[str, Any]:
        """
        导出系统状态

        Returns:
            系统状态数据
        """
        return {
            "config": {
                "population_size": self.config.population_size,
                "mutation_rate": self.config.mutation_rate,
                "thinking_rounds": self.config.thinking_rounds,
                "safety_threshold": self.config.safety_threshold
            },
            "status": self.get_system_status(),
            "insights": self.memory.export_insights(),
            "tools": self.tool_system.export_tools(),
            "reward_history": self.reward_system.export_reward_history(),
            "evolution_history": self.laws.export_history()
        }

    def run_evolution_cycle(
        self,
        cycles: int = 1,
        problems: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        运行多个进化循环

        Args:
            cycles: 循环次数
            problems: 问题列表

        Returns:
            每个循环的结果
        """
        results = []

        for i in range(cycles):
            problem = problems[i % len(problems)] if problems else None
            result = self.evolve(problem)
            results.append(result)

        return results


def create_neural_evolution_system(
    population_size: int = 10,
    thinking_rounds: int = 3,
    safety_threshold: float = 0.8
) -> NeuralEvolutionSystem:
    """
    创建神经进化系统的便捷函数

    Args:
        population_size: 种群大小
        thinking_rounds: 思考轮数
        safety_threshold: 安全阈值

    Returns:
        配置好的神经进化系统
    """
    config = EvolutionConfig(
        population_size=population_size,
        thinking_rounds=thinking_rounds,
        safety_threshold=safety_threshold
    )

    system = NeuralEvolutionSystem(config)
    system.initialize()

    return system
