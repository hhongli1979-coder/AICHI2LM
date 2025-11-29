# -*- coding: utf-8 -*-
"""
自我奖励系统 - Self-Rewarding System

实现内部评判和自我强化学习机制。
This implements internal evaluation and self-reinforcement learning mechanisms.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class RewardType(Enum):
    """奖励类型"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class Reward:
    """奖励记录"""
    reward_type: RewardType
    value: float
    reason: str
    task_type: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}


@dataclass
class EvaluationCriteria:
    """评估标准"""
    name: str
    description: str
    weight: float = 1.0
    threshold: float = 0.5


@dataclass
class PerformanceMetrics:
    """性能指标"""
    accuracy: float = 0.0
    completeness: float = 0.0
    coherence: float = 0.0
    creativity: float = 0.0
    efficiency: float = 0.0
    safety: float = 0.0

    def overall_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """计算综合得分"""
        if weights is None:
            weights = {
                "accuracy": 0.25,
                "completeness": 0.2,
                "coherence": 0.2,
                "creativity": 0.1,
                "efficiency": 0.1,
                "safety": 0.15
            }

        score = (
            self.accuracy * weights.get("accuracy", 0.2) +
            self.completeness * weights.get("completeness", 0.2) +
            self.coherence * weights.get("coherence", 0.2) +
            self.creativity * weights.get("creativity", 0.1) +
            self.efficiency * weights.get("efficiency", 0.1) +
            self.safety * weights.get("safety", 0.2)
        )
        return score


class SelfRewardingSystem:
    """
    自我奖励系统 - Self-Rewarding System

    实现内部评判机制，对自身表现进行评分，
    并基于评分进行自我强化学习。
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        reward_threshold: float = 0.7
    ):
        """
        初始化自我奖励系统

        Args:
            learning_rate: 学习率
            discount_factor: 折扣因子
            reward_threshold: 奖励阈值
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.reward_threshold = reward_threshold

        self.reward_history: List[Reward] = []
        self.cumulative_reward: float = 0.0

        self.evaluation_criteria = self._initialize_criteria()
        self.value_estimates: Dict[str, float] = {}

        self._statistics = {
            "total_evaluations": 0,
            "positive_rewards": 0,
            "negative_rewards": 0,
            "avg_reward": 0.0,
            "total_reinforcements": 0
        }

    def _initialize_criteria(self) -> Dict[str, EvaluationCriteria]:
        """
        初始化评估标准

        Returns:
            评估标准字典
        """
        return {
            "accuracy": EvaluationCriteria(
                name="accuracy",
                description="答案的准确性",
                weight=0.25,
                threshold=0.7
            ),
            "completeness": EvaluationCriteria(
                name="completeness",
                description="回答的完整性",
                weight=0.2,
                threshold=0.6
            ),
            "coherence": EvaluationCriteria(
                name="coherence",
                description="逻辑连贯性",
                weight=0.2,
                threshold=0.7
            ),
            "creativity": EvaluationCriteria(
                name="creativity",
                description="创新性和独特性",
                weight=0.1,
                threshold=0.5
            ),
            "efficiency": EvaluationCriteria(
                name="efficiency",
                description="执行效率",
                weight=0.1,
                threshold=0.6
            ),
            "safety": EvaluationCriteria(
                name="safety",
                description="安全性和合规性",
                weight=0.15,
                threshold=0.9
            )
        }

    def internal_judge(self, task: str, solution: str) -> Tuple[float, PerformanceMetrics]:
        """
        内部评判 - 对任务和解决方案进行评分

        Args:
            task: 任务描述
            solution: 解决方案

        Returns:
            评分和性能指标
        """
        metrics = self._evaluate_metrics(task, solution)
        score = metrics.overall_score(
            {c.name: c.weight for c in self.evaluation_criteria.values()}
        )

        self._statistics["total_evaluations"] += 1

        return score, metrics

    def _evaluate_metrics(self, task: str, solution: str) -> PerformanceMetrics:
        """
        评估各项指标

        Args:
            task: 任务
            solution: 解决方案

        Returns:
            性能指标
        """
        # 在实际实现中，这里应该使用更复杂的评估逻辑
        # 这里提供模拟评估

        metrics = PerformanceMetrics()

        # 准确性评估（基于解决方案长度和内容）
        metrics.accuracy = min(0.5 + len(solution) / 1000, 0.95)

        # 完整性评估
        metrics.completeness = 0.6
        if "结论" in solution or "conclusion" in solution.lower():
            metrics.completeness += 0.2
        if "步骤" in solution or "step" in solution.lower():
            metrics.completeness += 0.1

        # 连贯性评估
        metrics.coherence = 0.7
        if len(solution.split('\n')) > 3:
            metrics.coherence += 0.1

        # 创新性评估
        metrics.creativity = 0.5

        # 效率评估
        metrics.efficiency = 0.7

        # 安全性评估（检查敏感内容）
        metrics.safety = 0.95
        sensitive_words = ["危险", "非法", "攻击", "伤害"]
        for word in sensitive_words:
            if word in solution:
                metrics.safety -= 0.2

        return metrics

    def evaluate_own_performance(
        self,
        task: str,
        solution: str
    ) -> Tuple[float, Reward]:
        """
        评估自身表现并生成奖励

        Args:
            task: 任务
            solution: 解决方案

        Returns:
            评分和奖励
        """
        score, metrics = self.internal_judge(task, solution)

        # 确定奖励类型
        if score >= self.reward_threshold:
            reward_type = RewardType.POSITIVE
            reward_value = score
            self._statistics["positive_rewards"] += 1
        elif score >= self.reward_threshold * 0.7:
            reward_type = RewardType.NEUTRAL
            reward_value = 0.0
        else:
            reward_type = RewardType.NEGATIVE
            reward_value = -(self.reward_threshold - score)
            self._statistics["negative_rewards"] += 1

        reward = Reward(
            reward_type=reward_type,
            value=reward_value,
            reason=self._generate_reward_reason(score, metrics),
            task_type=self._extract_task_type(task),
            metadata={
                "metrics": {
                    "accuracy": metrics.accuracy,
                    "completeness": metrics.completeness,
                    "coherence": metrics.coherence,
                    "creativity": metrics.creativity,
                    "efficiency": metrics.efficiency,
                    "safety": metrics.safety
                }
            }
        )

        self.reward_history.append(reward)
        self.cumulative_reward += reward_value

        # 更新平均奖励
        self._statistics["avg_reward"] = (
            sum(r.value for r in self.reward_history) / len(self.reward_history)
        )

        return score, reward

    def _generate_reward_reason(
        self,
        score: float,
        metrics: PerformanceMetrics
    ) -> str:
        """
        生成奖励原因说明

        Args:
            score: 总分
            metrics: 性能指标

        Returns:
            原因说明
        """
        reasons = []

        if metrics.accuracy >= 0.8:
            reasons.append("高准确性")
        elif metrics.accuracy < 0.5:
            reasons.append("准确性需要提升")

        if metrics.completeness >= 0.8:
            reasons.append("回答完整")
        elif metrics.completeness < 0.5:
            reasons.append("回答不够完整")

        if metrics.safety >= 0.9:
            reasons.append("安全性良好")
        elif metrics.safety < 0.7:
            reasons.append("存在安全隐患")

        return "; ".join(reasons) if reasons else f"综合得分: {score:.2f}"

    def _extract_task_type(self, task: str) -> str:
        """
        从任务描述中提取任务类型

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
            "计算": "math",
            "数学": "math",
            "分析": "analysis",
            "总结": "summarization"
        }

        for keyword, task_type in keywords.items():
            if keyword in task:
                return task_type

        return "general"

    def reinforce_learning(
        self,
        reward: Reward,
        state_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        基于奖励进行强化学习更新

        Args:
            reward: 奖励信息
            state_key: 状态键（用于值函数更新）

        Returns:
            更新信息
        """
        state_key = state_key or reward.task_type

        # 更新值估计
        old_value = self.value_estimates.get(state_key, 0.0)
        td_error = reward.value - old_value
        new_value = old_value + self.learning_rate * td_error

        self.value_estimates[state_key] = new_value

        self._statistics["total_reinforcements"] += 1

        return {
            "state_key": state_key,
            "old_value": old_value,
            "new_value": new_value,
            "td_error": td_error,
            "reward": reward.value
        }

    def get_expected_reward(self, task_type: str) -> float:
        """
        获取任务类型的期望奖励

        Args:
            task_type: 任务类型

        Returns:
            期望奖励值
        """
        return self.value_estimates.get(task_type, 0.0)

    def adapt_criteria(self) -> Dict[str, float]:
        """
        根据历史表现自适应调整评估标准

        Returns:
            调整后的标准权重
        """
        if len(self.reward_history) < 10:
            return {c.name: c.weight for c in self.evaluation_criteria.values()}

        # 分析最近的奖励历史
        recent_rewards = self.reward_history[-100:]

        # 统计每个任务类型的表现
        task_performance: Dict[str, List[float]] = {}
        for r in recent_rewards:
            if r.task_type not in task_performance:
                task_performance[r.task_type] = []
            task_performance[r.task_type].append(r.value)

        # 根据低表现领域调整权重
        adjustments = {}
        for task_type, values in task_performance.items():
            avg = sum(values) / len(values) if values else 0
            if avg < 0:
                # 低表现领域，提高相关标准权重
                if task_type == "coding":
                    adjustments["accuracy"] = 0.02
                elif task_type == "writing":
                    adjustments["coherence"] = 0.02

        # 应用调整
        for criterion_name, adjustment in adjustments.items():
            if criterion_name in self.evaluation_criteria:
                self.evaluation_criteria[criterion_name].weight += adjustment

        # 归一化权重
        total_weight = sum(c.weight for c in self.evaluation_criteria.values())
        for criterion in self.evaluation_criteria.values():
            criterion.weight /= total_weight

        return {c.name: c.weight for c in self.evaluation_criteria.values()}

    def get_improvement_suggestions(
        self,
        metrics: PerformanceMetrics
    ) -> List[str]:
        """
        基于指标获取改进建议

        Args:
            metrics: 性能指标

        Returns:
            改进建议列表
        """
        suggestions = []

        for name, criterion in self.evaluation_criteria.items():
            metric_value = getattr(metrics, name, 0.0)
            if metric_value < criterion.threshold:
                suggestions.append(
                    f"提升{criterion.description}（当前: {metric_value:.2f}, 目标: {criterion.threshold:.2f}）"
                )

        return suggestions

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取系统统计信息

        Returns:
            统计信息
        """
        return {
            **self._statistics,
            "cumulative_reward": self.cumulative_reward,
            "reward_history_size": len(self.reward_history),
            "value_estimates": dict(self.value_estimates),
            "criteria_weights": {
                c.name: c.weight for c in self.evaluation_criteria.values()
            }
        }

    def reset_history(self, keep_value_estimates: bool = True) -> None:
        """
        重置历史记录

        Args:
            keep_value_estimates: 是否保留值估计
        """
        self.reward_history.clear()
        self.cumulative_reward = 0.0

        if not keep_value_estimates:
            self.value_estimates.clear()

        self._statistics = {
            "total_evaluations": 0,
            "positive_rewards": 0,
            "negative_rewards": 0,
            "avg_reward": 0.0,
            "total_reinforcements": 0
        }

    def export_reward_history(self) -> List[Dict[str, Any]]:
        """
        导出奖励历史

        Returns:
            奖励记录列表
        """
        return [
            {
                "reward_type": r.reward_type.value,
                "value": r.value,
                "reason": r.reason,
                "task_type": r.task_type,
                "timestamp": r.timestamp,
                "metadata": r.metadata
            }
            for r in self.reward_history
        ]
