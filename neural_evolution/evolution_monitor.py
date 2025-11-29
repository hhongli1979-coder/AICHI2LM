# -*- coding: utf-8 -*-
"""
进化监控系统 - Evolution Monitor System

实现进化过程的实时监控和策略调整。
This implements real-time monitoring and strategy adaptation for evolution processes.
"""

import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class MetricType(Enum):
    """指标类型"""
    INTELLIGENCE = "intelligence"
    LEARNING_SPEED = "learning_speed"
    CREATIVITY = "creativity"
    PROBLEM_SOLVING = "problem_solving"
    MEMORY = "memory"
    ADAPTATION = "adaptation"


@dataclass
class MetricRecord:
    """指标记录"""
    metric_type: MetricType
    value: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}


@dataclass
class EvolutionStrategy:
    """进化策略"""
    name: str
    parameters: Dict[str, float]
    performance: float = 0.0
    usage_count: int = 0
    last_used: float = field(default_factory=time.time)


@dataclass
class Alert:
    """警报"""
    level: str  # info, warning, critical
    message: str
    metric_type: Optional[MetricType] = None
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False


class EvolutionMonitor:
    """
    进化监控系统 - Evolution Monitor

    跟踪进化过程的各项指标，实时调整进化策略。
    Tracks evolution metrics and adapts strategies in real-time.
    """

    def __init__(
        self,
        history_size: int = 1000,
        alert_threshold: float = 0.3,
        adaptation_interval: int = 10
    ):
        """
        初始化进化监控系统

        Args:
            history_size: 历史记录最大容量
            alert_threshold: 警报阈值
            adaptation_interval: 策略调整间隔
        """
        self.history_size = history_size
        self.alert_threshold = alert_threshold
        self.adaptation_interval = adaptation_interval

        self.metrics_history: Dict[MetricType, List[MetricRecord]] = {
            metric_type: [] for metric_type in MetricType
        }

        self.current_strategy: Optional[EvolutionStrategy] = None
        self.available_strategies: Dict[str, EvolutionStrategy] = {}
        self.alerts: List[Alert] = []

        self._evaluation_count: int = 0
        self._last_adaptation: float = time.time()

        # 初始化默认策略
        self._initialize_strategies()

    def _initialize_strategies(self) -> None:
        """初始化默认进化策略"""
        self.available_strategies = {
            "balanced": EvolutionStrategy(
                name="balanced",
                parameters={
                    "mutation_rate": 0.1,
                    "crossover_rate": 0.7,
                    "elite_ratio": 0.2,
                    "exploration_factor": 0.5
                }
            ),
            "exploration": EvolutionStrategy(
                name="exploration",
                parameters={
                    "mutation_rate": 0.3,
                    "crossover_rate": 0.5,
                    "elite_ratio": 0.1,
                    "exploration_factor": 0.8
                }
            ),
            "exploitation": EvolutionStrategy(
                name="exploitation",
                parameters={
                    "mutation_rate": 0.05,
                    "crossover_rate": 0.8,
                    "elite_ratio": 0.3,
                    "exploration_factor": 0.2
                }
            ),
            "aggressive": EvolutionStrategy(
                name="aggressive",
                parameters={
                    "mutation_rate": 0.2,
                    "crossover_rate": 0.6,
                    "elite_ratio": 0.15,
                    "exploration_factor": 0.6
                }
            )
        }
        self.current_strategy = self.available_strategies["balanced"]

    def calculate_iq(self) -> float:
        """
        计算智商指标 - Intelligence Quotient

        Returns:
            IQ分数 (0-1)
        """
        # 基于各项能力指标计算综合IQ
        metrics = self.get_latest_metrics()

        if not metrics:
            return 0.5  # 默认值

        weights = {
            MetricType.PROBLEM_SOLVING: 0.3,
            MetricType.CREATIVITY: 0.2,
            MetricType.MEMORY: 0.2,
            MetricType.LEARNING_SPEED: 0.2,
            MetricType.ADAPTATION: 0.1
        }

        iq = sum(
            metrics.get(m_type, 0.5) * weight
            for m_type, weight in weights.items()
        )

        return iq

    def measure_learning_rate(self) -> float:
        """
        测量学习速度

        Returns:
            学习速度分数 (0-1)
        """
        history = self.metrics_history.get(MetricType.LEARNING_SPEED, [])

        if len(history) < 2:
            return 0.5

        # 计算最近的学习速度趋势
        recent = history[-10:]
        if len(recent) < 2:
            return recent[-1].value if recent else 0.5

        # 计算改进率
        improvements = []
        for i in range(1, len(recent)):
            diff = recent[i].value - recent[i - 1].value
            improvements.append(diff)

        avg_improvement = sum(improvements) / len(improvements) if improvements else 0

        # 转换为0-1范围
        return min(max(0.5 + avg_improvement * 5, 0.0), 1.0)

    def assess_creativity(self) -> float:
        """
        评估创造力

        Returns:
            创造力分数 (0-1)
        """
        history = self.metrics_history.get(MetricType.CREATIVITY, [])

        if not history:
            return 0.5

        # 使用最近记录的加权平均
        recent = history[-20:]
        if not recent:
            return 0.5

        # 较新的记录权重更高
        weights = [i / sum(range(1, len(recent) + 1)) for i in range(1, len(recent) + 1)]
        creativity = sum(r.value * w for r, w in zip(recent, weights))

        return creativity

    def evaluate_reasoning(self) -> float:
        """
        评估推理能力

        Returns:
            推理深度分数 (0-1)
        """
        history = self.metrics_history.get(MetricType.PROBLEM_SOLVING, [])

        if not history:
            return 0.5

        recent = history[-10:]
        return sum(r.value for r in recent) / len(recent)

    def track_evolution(self) -> Dict[str, Any]:
        """
        跟踪进化状态，返回综合指标

        Returns:
            指标字典
        """
        metrics = {
            "intelligence_quotient": self.calculate_iq(),
            "learning_speed": self.measure_learning_rate(),
            "creativity_score": self.assess_creativity(),
            "problem_solving_depth": self.evaluate_reasoning()
        }

        self._evaluation_count += 1

        # 检查是否需要调整策略
        if self._evaluation_count % self.adaptation_interval == 0:
            self.adapt_evolution_strategy(metrics)

        # 检查是否需要发出警报
        self._check_alerts(metrics)

        return metrics

    def record_metric(
        self,
        metric_type: MetricType,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        记录指标

        Args:
            metric_type: 指标类型
            value: 指标值
            metadata: 元数据
        """
        record = MetricRecord(
            metric_type=metric_type,
            value=value,
            metadata=metadata or {}
        )

        self.metrics_history[metric_type].append(record)

        # 限制历史记录大小
        if len(self.metrics_history[metric_type]) > self.history_size:
            self.metrics_history[metric_type] = self.metrics_history[metric_type][-self.history_size:]

    def get_latest_metrics(self) -> Dict[MetricType, float]:
        """
        获取最新指标

        Returns:
            最新指标字典
        """
        return {
            m_type: records[-1].value if records else 0.5
            for m_type, records in self.metrics_history.items()
        }

    def adapt_evolution_strategy(self, metrics: Dict[str, Any]) -> EvolutionStrategy:
        """
        根据指标自适应调整进化策略

        Args:
            metrics: 当前指标

        Returns:
            新策略
        """
        iq = metrics.get("intelligence_quotient", 0.5)
        learning_speed = metrics.get("learning_speed", 0.5)

        # 选择策略
        if iq < 0.4 and learning_speed < 0.4:
            # 表现差，需要更多探索
            new_strategy_name = "exploration"
        elif iq > 0.7 and learning_speed > 0.6:
            # 表现好，可以更多利用
            new_strategy_name = "exploitation"
        elif iq > 0.5:
            # 中等表现，使用平衡策略
            new_strategy_name = "balanced"
        else:
            # 需要激进改进
            new_strategy_name = "aggressive"

        new_strategy = self.available_strategies[new_strategy_name]

        # 更新当前策略
        if self.current_strategy:
            self.current_strategy.performance = iq

        self.current_strategy = new_strategy
        new_strategy.usage_count += 1
        new_strategy.last_used = time.time()

        self._last_adaptation = time.time()

        return new_strategy

    def _check_alerts(self, metrics: Dict[str, Any]) -> None:
        """
        检查并发出警报

        Args:
            metrics: 当前指标
        """
        for name, value in metrics.items():
            if value < self.alert_threshold:
                alert = Alert(
                    level="warning" if value > self.alert_threshold * 0.5 else "critical",
                    message=f"{name} is below threshold: {value:.2f} < {self.alert_threshold}",
                    metric_type=MetricType.INTELLIGENCE if "intelligence" in name else None
                )
                self.alerts.append(alert)

    def get_alerts(self, unresolved_only: bool = True) -> List[Alert]:
        """
        获取警报列表

        Args:
            unresolved_only: 是否只返回未解决的警报

        Returns:
            警报列表
        """
        if unresolved_only:
            return [a for a in self.alerts if not a.resolved]
        return self.alerts

    def resolve_alert(self, alert_index: int) -> bool:
        """
        解决警报

        Args:
            alert_index: 警报索引

        Returns:
            是否成功
        """
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index].resolved = True
            return True
        return False

    def get_trend(
        self,
        metric_type: MetricType,
        window: int = 10
    ) -> Dict[str, Any]:
        """
        获取指标趋势

        Args:
            metric_type: 指标类型
            window: 窗口大小

        Returns:
            趋势信息
        """
        history = self.metrics_history.get(metric_type, [])

        if len(history) < 2:
            return {"trend": "insufficient_data", "change": 0.0}

        recent = history[-window:]
        if len(recent) < 2:
            return {"trend": "insufficient_data", "change": 0.0}

        first_avg = sum(r.value for r in recent[:len(recent) // 2]) / (len(recent) // 2)
        second_avg = sum(r.value for r in recent[len(recent) // 2:]) / (len(recent) - len(recent) // 2)

        change = second_avg - first_avg

        if change > 0.05:
            trend = "improving"
        elif change < -0.05:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "change": change,
            "first_avg": first_avg,
            "second_avg": second_avg,
            "data_points": len(recent)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取监控系统统计

        Returns:
            统计信息
        """
        return {
            "evaluation_count": self._evaluation_count,
            "current_strategy": self.current_strategy.name if self.current_strategy else None,
            "metrics_counts": {
                m_type.value: len(records)
                for m_type, records in self.metrics_history.items()
            },
            "active_alerts": len([a for a in self.alerts if not a.resolved]),
            "total_alerts": len(self.alerts),
            "last_adaptation": self._last_adaptation,
            "strategy_usage": {
                name: s.usage_count
                for name, s in self.available_strategies.items()
            }
        }

    def get_strategy_parameters(self) -> Dict[str, float]:
        """
        获取当前策略参数

        Returns:
            策略参数
        """
        if self.current_strategy:
            return dict(self.current_strategy.parameters)
        return {}

    def register_strategy(
        self,
        name: str,
        parameters: Dict[str, float]
    ) -> EvolutionStrategy:
        """
        注册新策略

        Args:
            name: 策略名称
            parameters: 策略参数

        Returns:
            创建的策略
        """
        strategy = EvolutionStrategy(
            name=name,
            parameters=parameters
        )
        self.available_strategies[name] = strategy
        return strategy

    def export_history(
        self,
        metric_type: Optional[MetricType] = None
    ) -> List[Dict[str, Any]]:
        """
        导出历史记录

        Args:
            metric_type: 指标类型（None表示全部）

        Returns:
            历史记录列表
        """
        if metric_type:
            records = self.metrics_history.get(metric_type, [])
            return [
                {
                    "metric_type": metric_type.value,
                    "value": r.value,
                    "timestamp": r.timestamp,
                    "metadata": r.metadata
                }
                for r in records
            ]

        all_records = []
        for m_type, records in self.metrics_history.items():
            for r in records:
                all_records.append({
                    "metric_type": m_type.value,
                    "value": r.value,
                    "timestamp": r.timestamp,
                    "metadata": r.metadata
                })

        return sorted(all_records, key=lambda x: x["timestamp"])

    def clear_history(self) -> None:
        """清空历史记录"""
        for m_type in self.metrics_history:
            self.metrics_history[m_type] = []
        self.alerts.clear()
        self._evaluation_count = 0
