# -*- coding: utf-8 -*-
"""
多轮思考训练系统 - Multi-Round Thinking Training System

实现多轮深度思考和自我训练机制。
This implements multi-round deep thinking and self-training mechanisms.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ThinkingQuality(Enum):
    """思考质量等级"""
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


@dataclass
class Solution:
    """解决方案"""
    content: str
    round_number: int
    quality: ThinkingQuality
    score: float
    reasoning_steps: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.reasoning_steps:
            self.reasoning_steps = []
        if not self.improvements:
            self.improvements = []
        if not self.metadata:
            self.metadata = {}


@dataclass
class ThinkingStrategy:
    """思考策略"""
    name: str
    description: str
    weight: float = 1.0
    success_count: int = 0
    total_count: int = 0

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_count if self.total_count > 0 else 0.0


@dataclass
class TrainingData:
    """自生成训练数据"""
    question: str
    solution: str
    score: float
    source: str = "self_generated"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}


class MultiRoundThinking:
    """
    多轮思考训练系统 - Multi-Round Thinking Training System

    实现多轮深度思考，每轮基于前一轮结果进行深度反思和改进。
    Implements multi-round deep thinking with reflection and improvement
    based on previous rounds.
    """

    def __init__(
        self,
        thinking_rounds: int = 3,
        quality_threshold: float = 0.7,
        improvement_threshold: float = 0.05
    ):
        """
        初始化多轮思考系统

        Args:
            thinking_rounds: 默认思考轮数
            quality_threshold: 质量阈值
            improvement_threshold: 改进阈值（低于此值停止迭代）
        """
        self.thinking_rounds = thinking_rounds
        self.quality_threshold = quality_threshold
        self.improvement_threshold = improvement_threshold

        self.strategies: Dict[str, ThinkingStrategy] = {
            "analytical": ThinkingStrategy(
                name="analytical",
                description="分析问题并分解为子问题"
            ),
            "creative": ThinkingStrategy(
                name="creative",
                description="从创新角度思考问题"
            ),
            "critical": ThinkingStrategy(
                name="critical",
                description="批判性审视现有解决方案"
            ),
            "systematic": ThinkingStrategy(
                name="systematic",
                description="系统性地逐步推理"
            )
        }

        self._thinking_history: List[Dict[str, Any]] = []
        self._statistics = {
            "total_problems": 0,
            "total_rounds": 0,
            "avg_rounds_per_problem": 0.0,
            "avg_improvement": 0.0
        }

    def deep_think(
        self,
        problem: str,
        previous_solution: Optional[Solution] = None,
        strategy: Optional[str] = None
    ) -> Solution:
        """
        深度思考 - 执行一轮深度思考

        Args:
            problem: 问题描述
            previous_solution: 前一轮的解决方案
            strategy: 使用的思考策略

        Returns:
            本轮解决方案
        """
        round_number = (previous_solution.round_number + 1) if previous_solution else 1
        strategy_name = strategy or self._select_strategy(previous_solution)

        # 生成推理步骤
        reasoning_steps = self._generate_reasoning(
            problem,
            previous_solution,
            strategy_name
        )

        # 生成解决方案内容
        content = self._generate_solution(
            problem,
            reasoning_steps,
            previous_solution
        )

        # 评估质量
        score = self._evaluate_solution(content, problem)
        quality = self._score_to_quality(score)

        # 识别改进点
        improvements = self._identify_improvements(
            content,
            previous_solution
        )

        solution = Solution(
            content=content,
            round_number=round_number,
            quality=quality,
            score=score,
            reasoning_steps=reasoning_steps,
            improvements=improvements,
            metadata={
                "strategy": strategy_name,
                "problem_hash": hash(problem) % 10000
            }
        )

        # 更新策略统计
        self._update_strategy_stats(strategy_name, score)

        return solution

    def _select_strategy(self, previous_solution: Optional[Solution]) -> str:
        """
        选择思考策略

        Args:
            previous_solution: 前一轮解决方案

        Returns:
            策略名称
        """
        if previous_solution is None:
            # 第一轮使用分析策略
            return "analytical"

        # 根据前一轮质量选择策略
        if previous_solution.quality == ThinkingQuality.POOR:
            return "systematic"
        elif previous_solution.quality == ThinkingQuality.FAIR:
            return "analytical"
        else:
            # 质量较好时使用批判性思考改进
            return "critical"

    def _generate_reasoning(
        self,
        problem: str,
        previous: Optional[Solution],
        strategy: str
    ) -> List[str]:
        """
        生成推理步骤

        Args:
            problem: 问题
            previous: 前一轮解决方案
            strategy: 策略名称

        Returns:
            推理步骤列表
        """
        steps = []

        if strategy == "analytical":
            steps = [
                f"1. 分析问题: {problem[:50]}...",
                "2. 识别关键要素",
                "3. 分解为子问题",
                "4. 逐个解决子问题",
                "5. 整合答案"
            ]
        elif strategy == "creative":
            steps = [
                "1. 发散思维探索可能性",
                "2. 类比其他领域的解决方案",
                "3. 组合不同思路",
                "4. 评估创新方案可行性"
            ]
        elif strategy == "critical":
            steps = [
                "1. 审视当前解决方案的假设",
                "2. 识别潜在缺陷",
                "3. 提出反例测试",
                "4. 修正和完善"
            ]
        elif strategy == "systematic":
            steps = [
                "1. 明确问题边界",
                "2. 建立解决框架",
                "3. 按步骤执行",
                "4. 验证每一步结果"
            ]

        # 如果有前一轮结果，添加反思步骤
        if previous:
            steps.append(f"反思: 前一轮得分 {previous.score:.2f}，需要改进")
            for imp in previous.improvements[:2]:
                steps.append(f"改进点: {imp}")

        return steps

    def _generate_solution(
        self,
        problem: str,
        reasoning_steps: List[str],
        previous: Optional[Solution]
    ) -> str:
        """
        生成解决方案内容

        Args:
            problem: 问题
            reasoning_steps: 推理步骤
            previous: 前一轮解决方案

        Returns:
            解决方案内容
        """
        # 在实际实现中，这里应该调用语言模型生成
        # 这里提供模拟实现
        solution_parts = []

        solution_parts.append(f"问题分析: {problem}")
        solution_parts.append("推理过程:")
        for step in reasoning_steps:
            solution_parts.append(f"  {step}")

        if previous:
            solution_parts.append(f"基于前一轮改进: {previous.improvements[0] if previous.improvements else '无'}")

        solution_parts.append("结论: [Generated Solution]")

        return "\n".join(solution_parts)

    def _evaluate_solution(self, solution: str, problem: str) -> float:
        """
        评估解决方案质量

        Args:
            solution: 解决方案
            problem: 原问题

        Returns:
            质量分数 (0-1)
        """
        # 在实际实现中，应该使用更复杂的评估机制
        # 这里提供模拟评估
        score = 0.5

        # 检查是否包含推理过程
        if "推理过程" in solution:
            score += 0.1

        # 检查是否有结论
        if "结论" in solution:
            score += 0.1

        # 检查长度（简单启发式）
        if len(solution) > 100:
            score += 0.1
        if len(solution) > 300:
            score += 0.1

        return min(score, 1.0)

    def _score_to_quality(self, score: float) -> ThinkingQuality:
        """
        将分数转换为质量等级

        Args:
            score: 分数

        Returns:
            质量等级
        """
        if score >= 0.9:
            return ThinkingQuality.EXCELLENT
        elif score >= 0.7:
            return ThinkingQuality.GOOD
        elif score >= 0.5:
            return ThinkingQuality.FAIR
        else:
            return ThinkingQuality.POOR

    def _identify_improvements(
        self,
        solution: str,
        previous: Optional[Solution]
    ) -> List[str]:
        """
        识别改进点

        Args:
            solution: 当前解决方案
            previous: 前一轮解决方案

        Returns:
            改进建议列表
        """
        improvements = []

        # 基础改进建议
        if len(solution) < 200:
            improvements.append("增加更多细节和解释")

        if "例如" not in solution and "比如" not in solution:
            improvements.append("添加具体示例")

        if previous:
            if len(solution) <= len(previous.content):
                improvements.append("扩展解决方案的深度")

        return improvements

    def _update_strategy_stats(self, strategy_name: str, score: float) -> None:
        """
        更新策略统计

        Args:
            strategy_name: 策略名称
            score: 得分
        """
        if strategy_name in self.strategies:
            strategy = self.strategies[strategy_name]
            strategy.total_count += 1
            if score >= self.quality_threshold:
                strategy.success_count += 1

    def adjust_thinking_strategy(self, quality: ThinkingQuality) -> None:
        """
        根据质量调整思考策略权重

        Args:
            quality: 当前解决方案质量
        """
        # 根据质量调整策略权重
        if quality == ThinkingQuality.POOR:
            self.strategies["systematic"].weight *= 1.1
            self.strategies["analytical"].weight *= 1.05
        elif quality == ThinkingQuality.EXCELLENT:
            self.strategies["creative"].weight *= 1.1
            self.strategies["critical"].weight *= 1.05

    def train_self(self, problem: str) -> Tuple[Solution, List[Solution]]:
        """
        自我训练 - 执行多轮思考

        Args:
            problem: 问题

        Returns:
            最佳解决方案和所有解决方案列表
        """
        solutions = []
        self._statistics["total_problems"] += 1

        for round_num in range(self.thinking_rounds):
            previous = solutions[-1] if solutions else None

            # 执行一轮深度思考
            solution = self.deep_think(problem, previous)
            solutions.append(solution)

            self._statistics["total_rounds"] += 1

            # 调整策略
            self.adjust_thinking_strategy(solution.quality)

            # 检查是否达到质量阈值
            if solution.score >= self.quality_threshold:
                break

            # 检查改进是否足够
            if previous and (solution.score - previous.score) < self.improvement_threshold:
                break

        # 记录历史
        self._thinking_history.append({
            "problem": problem[:100],
            "rounds": len(solutions),
            "final_score": solutions[-1].score,
            "timestamp": time.time()
        })

        # 更新统计
        self._update_statistics()

        # 选择最佳解决方案
        best = self.select_best_solution(solutions)

        return best, solutions

    def select_best_solution(self, solutions: List[Solution]) -> Solution:
        """
        选择最佳解决方案

        Args:
            solutions: 解决方案列表

        Returns:
            最佳解决方案
        """
        if not solutions:
            raise ValueError("No solutions to select from")

        return max(solutions, key=lambda s: s.score)

    def _update_statistics(self) -> None:
        """更新统计信息"""
        if self._statistics["total_problems"] > 0:
            self._statistics["avg_rounds_per_problem"] = (
                self._statistics["total_rounds"] / self._statistics["total_problems"]
            )

        # 计算平均改进
        if len(self._thinking_history) > 0:
            improvements = []
            for record in self._thinking_history[-100:]:  # 只看最近100条
                if record["rounds"] > 1:
                    improvements.append(record["final_score"])
            if improvements:
                self._statistics["avg_improvement"] = sum(improvements) / len(improvements)

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取系统统计

        Returns:
            统计信息
        """
        strategy_stats = {
            name: {
                "weight": s.weight,
                "success_rate": s.success_rate,
                "total_count": s.total_count
            }
            for name, s in self.strategies.items()
        }

        return {
            **self._statistics,
            "strategies": strategy_stats,
            "thinking_rounds": self.thinking_rounds,
            "quality_threshold": self.quality_threshold,
            "history_size": len(self._thinking_history)
        }

    def generate_training_data(
        self,
        problems: List[str]
    ) -> List[TrainingData]:
        """
        生成训练数据 - 通过自我思考生成高质量训练数据

        Args:
            problems: 问题列表

        Returns:
            训练数据列表
        """
        training_data = []

        for problem in problems:
            best_solution, _ = self.train_self(problem)

            if best_solution.score >= self.quality_threshold:
                data = TrainingData(
                    question=problem,
                    solution=best_solution.content,
                    score=best_solution.score,
                    metadata={
                        "rounds": best_solution.round_number,
                        "strategy": best_solution.metadata.get("strategy", "unknown")
                    }
                )
                training_data.append(data)

        return training_data
