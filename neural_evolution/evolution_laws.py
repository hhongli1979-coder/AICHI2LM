# -*- coding: utf-8 -*-
"""
进化安全定律 - Evolutionary Safety Laws

实现进化过程的安全约束机制。
This implements safety constraints for evolution processes.

进化三定律 (Three Laws of Evolution):
1. 第一定律（Endure）: 保障系统安全稳定
2. 第二定律（Excel）: 保持或提升性能
3. 第三定律（Evolve）: 满足前两者后自主优化
"""

import time
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ModificationType(Enum):
    """修改类型"""
    PARAMETER_CHANGE = "parameter_change"
    ARCHITECTURE_CHANGE = "architecture_change"
    CAPABILITY_ADDITION = "capability_addition"
    CAPABILITY_REMOVAL = "capability_removal"
    BEHAVIOR_MODIFICATION = "behavior_modification"


class SafetyLevel(Enum):
    """安全级别"""
    SAFE = "safe"
    CAUTIOUS = "cautious"
    RISKY = "risky"
    DANGEROUS = "dangerous"


@dataclass
class Modification:
    """修改请求"""
    modification_id: str
    modification_type: ModificationType
    description: str
    changes: Dict[str, Any]
    expected_impact: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}


@dataclass
class SafetyCheckResult:
    """安全检查结果"""
    passed: bool
    safety_level: SafetyLevel
    reasons: List[str]
    score: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class PerformanceCheckResult:
    """性能检查结果"""
    passed: bool
    expected_change: float
    confidence: float
    reasons: List[str]
    timestamp: float = field(default_factory=time.time)


class EvolutionaryLaws:
    """
    进化安全定律 - Evolutionary Laws

    实现进化过程的三大安全定律，确保系统安全、稳定地进化。
    """

    def __init__(
        self,
        safety_threshold: float = 0.8,
        performance_threshold: float = 0.0,
        rollback_enabled: bool = True
    ):
        """
        初始化进化定律系统

        Args:
            safety_threshold: 安全阈值（0-1）
            performance_threshold: 性能阈值（允许的最大性能下降）
            rollback_enabled: 是否启用回滚
        """
        self.safety_threshold = safety_threshold
        self.performance_threshold = performance_threshold
        self.rollback_enabled = rollback_enabled

        self.modification_history: List[Tuple[Modification, bool]] = []
        self.current_performance: float = 0.5
        self.baseline_performance: float = 0.5

        self._safety_rules = self._initialize_safety_rules()
        self._blocked_modifications: List[Modification] = []

        self._statistics = {
            "total_checks": 0,
            "passed_checks": 0,
            "blocked_checks": 0,
            "rollbacks": 0
        }

    def _initialize_safety_rules(self) -> List[Dict[str, Any]]:
        """
        初始化安全规则

        Returns:
            安全规则列表
        """
        return [
            {
                "name": "no_harmful_content",
                "description": "禁止产生有害内容",
                "check": self._check_no_harmful_content,
                "severity": "critical"
            },
            {
                "name": "stability_preserved",
                "description": "保持系统稳定性",
                "check": self._check_stability,
                "severity": "high"
            },
            {
                "name": "reversibility",
                "description": "修改必须可逆",
                "check": self._check_reversibility,
                "severity": "medium"
            },
            {
                "name": "resource_limits",
                "description": "资源使用在限制内",
                "check": self._check_resource_limits,
                "severity": "medium"
            },
            {
                "name": "privacy_protection",
                "description": "保护隐私数据",
                "check": self._check_privacy,
                "severity": "critical"
            }
        ]

    def _check_no_harmful_content(self, modification: Modification) -> Tuple[bool, str]:
        """检查是否产生有害内容"""
        harmful_keywords = ["攻击", "伤害", "破坏", "非法", "恶意"]
        description = modification.description.lower()

        for keyword in harmful_keywords:
            if keyword in description:
                return False, f"检测到潜在有害关键词: {keyword}"

        return True, "未检测到有害内容"

    def _check_stability(self, modification: Modification) -> Tuple[bool, str]:
        """检查系统稳定性"""
        # 大规模架构变更可能影响稳定性
        if modification.modification_type == ModificationType.ARCHITECTURE_CHANGE:
            impact = modification.expected_impact
            if impact > 0.5:
                return False, "架构变更影响过大，可能影响稳定性"

        return True, "稳定性检查通过"

    def _check_reversibility(self, modification: Modification) -> Tuple[bool, str]:
        """检查可逆性"""
        # 检查修改是否包含回滚信息
        if "rollback_procedure" in modification.metadata:
            return True, "提供了回滚程序"

        # 某些类型的修改自动可逆
        if modification.modification_type == ModificationType.PARAMETER_CHANGE:
            return True, "参数修改可自动回滚"

        return False, "未提供回滚程序"

    def _check_resource_limits(self, modification: Modification) -> Tuple[bool, str]:
        """检查资源限制"""
        changes = modification.changes

        # 检查内存使用
        if "memory_increase" in changes:
            if changes["memory_increase"] > 0.5:  # 50% 增加
                return False, "内存增加超过限制"

        # 检查计算资源
        if "compute_increase" in changes:
            if changes["compute_increase"] > 0.3:  # 30% 增加
                return False, "计算资源增加超过限制"

        return True, "资源使用在限制内"

    def _check_privacy(self, modification: Modification) -> Tuple[bool, str]:
        """检查隐私保护"""
        changes = modification.changes

        # 检查是否涉及用户数据
        if "user_data_access" in changes:
            if changes["user_data_access"]:
                return False, "修改涉及用户数据访问"

        return True, "隐私检查通过"

    def law1_endure(self, modification: Modification) -> SafetyCheckResult:
        """
        第一定律：保障系统安全稳定

        确保任何修改都不会危害系统的安全性和稳定性。

        Args:
            modification: 待检查的修改

        Returns:
            安全检查结果
        """
        reasons = []
        failed_critical = False
        failed_any = False
        total_score = 0.0
        rule_count = len(self._safety_rules)

        for rule in self._safety_rules:
            passed, reason = rule["check"](modification)
            if not passed:
                failed_any = True
                reasons.append(f"[{rule['severity']}] {rule['name']}: {reason}")
                if rule["severity"] == "critical":
                    failed_critical = True
            else:
                total_score += 1.0 / rule_count

        # 确定安全级别
        if failed_critical:
            safety_level = SafetyLevel.DANGEROUS
        elif failed_any:
            safety_level = SafetyLevel.RISKY
        elif total_score >= self.safety_threshold:
            safety_level = SafetyLevel.SAFE
        else:
            safety_level = SafetyLevel.CAUTIOUS

        passed = safety_level in [SafetyLevel.SAFE, SafetyLevel.CAUTIOUS] and not failed_critical

        return SafetyCheckResult(
            passed=passed,
            safety_level=safety_level,
            reasons=reasons if reasons else ["所有安全检查通过"],
            score=total_score
        )

    def law2_excel(self, modification: Modification) -> PerformanceCheckResult:
        """
        第二定律：保持或提升性能

        确保修改不会导致性能显著下降。

        Args:
            modification: 待检查的修改

        Returns:
            性能检查结果
        """
        expected_change = modification.expected_impact
        reasons = []

        # 计算预期性能变化
        if expected_change < self.performance_threshold:
            passed = False
            reasons.append(f"预期性能下降 {-expected_change:.2%}，超过阈值 {-self.performance_threshold:.2%}")
        else:
            passed = True
            if expected_change > 0:
                reasons.append(f"预期性能提升 {expected_change:.2%}")
            else:
                reasons.append("性能保持稳定")

        # 计算置信度（基于修改类型）
        confidence = self._calculate_confidence(modification)

        return PerformanceCheckResult(
            passed=passed,
            expected_change=expected_change,
            confidence=confidence,
            reasons=reasons
        )

    def _calculate_confidence(self, modification: Modification) -> float:
        """
        计算性能预测置信度

        Args:
            modification: 修改

        Returns:
            置信度 (0-1)
        """
        # 参数修改置信度较高
        if modification.modification_type == ModificationType.PARAMETER_CHANGE:
            return 0.8

        # 架构修改置信度较低
        if modification.modification_type == ModificationType.ARCHITECTURE_CHANGE:
            return 0.4

        # 默认中等置信度
        return 0.6

    def law3_evolve(self, modification: Modification) -> Tuple[bool, Dict[str, Any]]:
        """
        第三定律：满足前两者后自主优化

        只有在满足安全和性能要求后，才允许进行进化优化。

        Args:
            modification: 待应用的修改

        Returns:
            是否允许进化及详细结果
        """
        self._statistics["total_checks"] += 1

        # 先检查第一定律（安全）
        safety_result = self.law1_endure(modification)
        if not safety_result.passed:
            self._statistics["blocked_checks"] += 1
            self._blocked_modifications.append(modification)
            return False, {
                "stage": "law1_endure",
                "passed": False,
                "safety_result": {
                    "passed": safety_result.passed,
                    "safety_level": safety_result.safety_level.value,
                    "reasons": safety_result.reasons,
                    "score": safety_result.score
                }
            }

        # 再检查第二定律（性能）
        performance_result = self.law2_excel(modification)
        if not performance_result.passed:
            self._statistics["blocked_checks"] += 1
            self._blocked_modifications.append(modification)
            return False, {
                "stage": "law2_excel",
                "passed": False,
                "performance_result": {
                    "passed": performance_result.passed,
                    "expected_change": performance_result.expected_change,
                    "confidence": performance_result.confidence,
                    "reasons": performance_result.reasons
                }
            }

        # 两个定律都通过，允许进化
        self._statistics["passed_checks"] += 1
        return True, {
            "stage": "law3_evolve",
            "passed": True,
            "safety_result": {
                "passed": safety_result.passed,
                "safety_level": safety_result.safety_level.value,
                "score": safety_result.score
            },
            "performance_result": {
                "passed": performance_result.passed,
                "expected_change": performance_result.expected_change,
                "confidence": performance_result.confidence
            }
        }

    def apply_evolution(
        self,
        modification: Modification,
        apply_func: Optional[Callable[[Modification], bool]] = None
    ) -> Dict[str, Any]:
        """
        应用进化修改（在三定律检查通过后）

        Args:
            modification: 修改
            apply_func: 实际应用修改的函数

        Returns:
            应用结果
        """
        # 首先进行三定律检查
        allowed, check_result = self.law3_evolve(modification)

        if not allowed:
            return {
                "success": False,
                "reason": "三定律检查未通过",
                "details": check_result
            }

        # 应用修改
        success = True
        if apply_func:
            try:
                success = apply_func(modification)
            except Exception as e:
                success = False
                return {
                    "success": False,
                    "reason": f"应用修改时出错: {str(e)}",
                    "details": check_result
                }

        # 记录历史
        self.modification_history.append((modification, success))

        return {
            "success": success,
            "reason": "进化修改成功应用" if success else "应用失败",
            "details": check_result
        }

    def rollback(self, steps: int = 1) -> Dict[str, Any]:
        """
        回滚修改

        Args:
            steps: 回滚步数

        Returns:
            回滚结果
        """
        if not self.rollback_enabled:
            return {"success": False, "reason": "回滚功能未启用"}

        if len(self.modification_history) < steps:
            return {"success": False, "reason": "历史记录不足"}

        rolled_back = []
        for _ in range(steps):
            if self.modification_history:
                mod, _ = self.modification_history.pop()
                rolled_back.append(mod.modification_id)

        self._statistics["rollbacks"] += len(rolled_back)

        return {
            "success": True,
            "rolled_back": rolled_back,
            "remaining_history": len(self.modification_history)
        }

    def get_blocked_modifications(self) -> List[Dict[str, Any]]:
        """
        获取被阻止的修改列表

        Returns:
            被阻止的修改
        """
        return [
            {
                "modification_id": m.modification_id,
                "type": m.modification_type.value,
                "description": m.description,
                "timestamp": m.timestamp
            }
            for m in self._blocked_modifications
        ]

    def update_performance(self, performance: float) -> None:
        """
        更新当前性能基准

        Args:
            performance: 新的性能值
        """
        self.current_performance = performance

    def set_baseline(self, performance: float) -> None:
        """
        设置性能基线

        Args:
            performance: 基线性能值
        """
        self.baseline_performance = performance

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            统计信息
        """
        return {
            **self._statistics,
            "safety_threshold": self.safety_threshold,
            "performance_threshold": self.performance_threshold,
            "current_performance": self.current_performance,
            "baseline_performance": self.baseline_performance,
            "modification_history_size": len(self.modification_history),
            "blocked_count": len(self._blocked_modifications)
        }

    def add_safety_rule(
        self,
        name: str,
        description: str,
        check_func: Callable[[Modification], Tuple[bool, str]],
        severity: str = "medium"
    ) -> None:
        """
        添加自定义安全规则

        Args:
            name: 规则名称
            description: 规则描述
            check_func: 检查函数
            severity: 严重程度 (critical, high, medium, low)
        """
        self._safety_rules.append({
            "name": name,
            "description": description,
            "check": check_func,
            "severity": severity
        })

    def export_history(self) -> List[Dict[str, Any]]:
        """
        导出修改历史

        Returns:
            历史记录
        """
        return [
            {
                "modification_id": mod.modification_id,
                "type": mod.modification_type.value,
                "description": mod.description,
                "success": success,
                "timestamp": mod.timestamp
            }
            for mod, success in self.modification_history
        ]
