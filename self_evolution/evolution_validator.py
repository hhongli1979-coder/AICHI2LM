"""
进化效果验证 (Evolution Validator)
=================================

验证进化改进效果:
- 性能改进测量
- 知识扩展评估
- 推理增强评估
- 效率改进计算
- 综合评估与回滚决策
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class ValidationMetricType(Enum):
    """验证指标类型"""
    PERFORMANCE = "performance"
    KNOWLEDGE = "knowledge"
    REASONING = "reasoning"
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"
    CREATIVITY = "creativity"


@dataclass
class ValidationMetric:
    """验证指标数据类"""
    metric_type: ValidationMetricType
    name: str
    old_value: float
    new_value: float
    weight: float = 1.0
    
    @property
    def improvement(self) -> float:
        """计算改进幅度"""
        if self.old_value == 0:
            return 0.0 if self.new_value == 0 else 1.0
        return (self.new_value - self.old_value) / abs(self.old_value)
        
    @property
    def is_improved(self) -> bool:
        """是否有改进"""
        return self.new_value > self.old_value


@dataclass
class ValidationResult:
    """验证结果数据类"""
    is_valid: bool
    overall_improvement: float
    metrics: List[ValidationMetric]
    recommendation: str  # 'commit', 'rollback', 'partial_commit'
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)
    validated_at: datetime = field(default_factory=datetime.now)


@dataclass
class EvolutionaryChange:
    """进化变更数据类"""
    change_id: str
    change_type: str  # 'parameter', 'architecture', 'knowledge', 'capability'
    description: str
    old_state: Dict[str, Any]
    new_state: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class EvolutionValidator:
    """
    进化效果验证器
    
    负责:
    1. 性能改进测量
    2. 知识扩展评估
    3. 推理增强评估
    4. 效率改进计算
    5. 综合评估与回滚决策
    """
    
    def __init__(self, improvement_threshold: float = 0.02):
        """
        初始化验证器
        
        Args:
            improvement_threshold: 改进阈值
        """
        self.improvement_threshold = improvement_threshold
        self.validation_history: List[ValidationResult] = []
        self.pending_changes: List[EvolutionaryChange] = []
        self.committed_changes: List[EvolutionaryChange] = []
        self.rolled_back_changes: List[EvolutionaryChange] = []
        
        # 指标权重配置
        self.metric_weights = {
            ValidationMetricType.PERFORMANCE: 1.2,
            ValidationMetricType.KNOWLEDGE: 1.0,
            ValidationMetricType.REASONING: 1.1,
            ValidationMetricType.EFFICIENCY: 0.9,
            ValidationMetricType.ROBUSTNESS: 1.0,
            ValidationMetricType.CREATIVITY: 0.8,
        }
        
    def measure_performance_gain(
        self,
        old_metrics: Dict[str, Any],
        new_metrics: Dict[str, Any]
    ) -> List[ValidationMetric]:
        """
        测量性能改进
        
        Args:
            old_metrics: 旧指标
            new_metrics: 新指标
            
        Returns:
            List[ValidationMetric]: 性能验证指标列表
        """
        performance_metrics = []
        
        # 准确率改进
        old_accuracy = old_metrics.get('accuracy', 0)
        new_accuracy = new_metrics.get('accuracy', 0)
        performance_metrics.append(ValidationMetric(
            metric_type=ValidationMetricType.PERFORMANCE,
            name='accuracy',
            old_value=old_accuracy,
            new_value=new_accuracy,
            weight=1.2
        ))
        
        # 损失降低
        old_loss = old_metrics.get('loss', 1.0)
        new_loss = new_metrics.get('loss', 1.0)
        # 对于损失，越低越好，所以反转计算
        performance_metrics.append(ValidationMetric(
            metric_type=ValidationMetricType.PERFORMANCE,
            name='loss_reduction',
            old_value=1.0 / (1.0 + old_loss),
            new_value=1.0 / (1.0 + new_loss),
            weight=1.0
        ))
        
        # 任务成功率
        old_success = old_metrics.get('task_success_rate', 0)
        new_success = new_metrics.get('task_success_rate', 0)
        performance_metrics.append(ValidationMetric(
            metric_type=ValidationMetricType.PERFORMANCE,
            name='task_success_rate',
            old_value=old_success,
            new_value=new_success,
            weight=1.1
        ))
        
        logger.info(f"Measured {len(performance_metrics)} performance metrics")
        return performance_metrics
        
    def assess_knowledge_growth(
        self,
        old_knowledge_state: Dict[str, Any],
        new_knowledge_state: Dict[str, Any]
    ) -> List[ValidationMetric]:
        """
        评估知识增长
        
        Args:
            old_knowledge_state: 旧知识状态
            new_knowledge_state: 新知识状态
            
        Returns:
            List[ValidationMetric]: 知识验证指标列表
        """
        knowledge_metrics = []
        
        # 知识覆盖范围
        old_domains = len(old_knowledge_state.get('domains', []))
        new_domains = len(new_knowledge_state.get('domains', []))
        knowledge_metrics.append(ValidationMetric(
            metric_type=ValidationMetricType.KNOWLEDGE,
            name='domain_coverage',
            old_value=old_domains,
            new_value=new_domains,
            weight=1.0
        ))
        
        # 知识深度
        old_depth = old_knowledge_state.get('knowledge_depth', 0)
        new_depth = new_knowledge_state.get('knowledge_depth', 0)
        knowledge_metrics.append(ValidationMetric(
            metric_type=ValidationMetricType.KNOWLEDGE,
            name='knowledge_depth',
            old_value=old_depth,
            new_value=new_depth,
            weight=1.1
        ))
        
        # 事实准确性
        old_factual = old_knowledge_state.get('factual_accuracy', 0)
        new_factual = new_knowledge_state.get('factual_accuracy', 0)
        knowledge_metrics.append(ValidationMetric(
            metric_type=ValidationMetricType.KNOWLEDGE,
            name='factual_accuracy',
            old_value=old_factual,
            new_value=new_factual,
            weight=1.2
        ))
        
        logger.info(f"Assessed {len(knowledge_metrics)} knowledge metrics")
        return knowledge_metrics
        
    def evaluate_reasoning_improvement(
        self,
        old_reasoning_metrics: Dict[str, Any],
        new_reasoning_metrics: Dict[str, Any]
    ) -> List[ValidationMetric]:
        """
        评估推理增强
        
        Args:
            old_reasoning_metrics: 旧推理指标
            new_reasoning_metrics: 新推理指标
            
        Returns:
            List[ValidationMetric]: 推理验证指标列表
        """
        reasoning_metrics = []
        
        # 逻辑一致性
        old_consistency = old_reasoning_metrics.get('logical_consistency', 0)
        new_consistency = new_reasoning_metrics.get('logical_consistency', 0)
        reasoning_metrics.append(ValidationMetric(
            metric_type=ValidationMetricType.REASONING,
            name='logical_consistency',
            old_value=old_consistency,
            new_value=new_consistency,
            weight=1.2
        ))
        
        # 多步推理能力
        old_multi_step = old_reasoning_metrics.get('multi_step_accuracy', 0)
        new_multi_step = new_reasoning_metrics.get('multi_step_accuracy', 0)
        reasoning_metrics.append(ValidationMetric(
            metric_type=ValidationMetricType.REASONING,
            name='multi_step_accuracy',
            old_value=old_multi_step,
            new_value=new_multi_step,
            weight=1.1
        ))
        
        # 推理深度
        old_depth = old_reasoning_metrics.get('reasoning_depth', 0)
        new_depth = new_reasoning_metrics.get('reasoning_depth', 0)
        reasoning_metrics.append(ValidationMetric(
            metric_type=ValidationMetricType.REASONING,
            name='reasoning_depth',
            old_value=old_depth,
            new_value=new_depth,
            weight=1.0
        ))
        
        logger.info(f"Evaluated {len(reasoning_metrics)} reasoning metrics")
        return reasoning_metrics
        
    def calculate_efficiency_improvement(
        self,
        old_efficiency_metrics: Dict[str, Any],
        new_efficiency_metrics: Dict[str, Any]
    ) -> List[ValidationMetric]:
        """
        计算效率改进
        
        Args:
            old_efficiency_metrics: 旧效率指标
            new_efficiency_metrics: 新效率指标
            
        Returns:
            List[ValidationMetric]: 效率验证指标列表
        """
        efficiency_metrics = []
        
        # 推理速度（tokens/秒）
        old_speed = old_efficiency_metrics.get('inference_speed', 0)
        new_speed = new_efficiency_metrics.get('inference_speed', 0)
        efficiency_metrics.append(ValidationMetric(
            metric_type=ValidationMetricType.EFFICIENCY,
            name='inference_speed',
            old_value=old_speed,
            new_value=new_speed,
            weight=1.0
        ))
        
        # 内存使用（越低越好，转换为效率分数）
        old_memory = old_efficiency_metrics.get('memory_usage_gb', 100)
        new_memory = new_efficiency_metrics.get('memory_usage_gb', 100)
        efficiency_metrics.append(ValidationMetric(
            metric_type=ValidationMetricType.EFFICIENCY,
            name='memory_efficiency',
            old_value=1.0 / (1.0 + old_memory),
            new_value=1.0 / (1.0 + new_memory),
            weight=0.8
        ))
        
        # 吞吐量
        old_throughput = old_efficiency_metrics.get('throughput', 0)
        new_throughput = new_efficiency_metrics.get('throughput', 0)
        efficiency_metrics.append(ValidationMetric(
            metric_type=ValidationMetricType.EFFICIENCY,
            name='throughput',
            old_value=old_throughput,
            new_value=new_throughput,
            weight=0.9
        ))
        
        logger.info(f"Calculated {len(efficiency_metrics)} efficiency metrics")
        return efficiency_metrics
        
    def comprehensive_evaluation(
        self,
        all_metrics: List[ValidationMetric]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        综合评估
        
        Args:
            all_metrics: 所有验证指标
            
        Returns:
            Tuple[float, Dict[str, Any]]: (综合改进分数, 详细分析)
        """
        if not all_metrics:
            return 0.0, {}
            
        # 按类型分组计算
        type_improvements = {}
        type_weights = {}
        
        for metric in all_metrics:
            metric_type = metric.metric_type
            improvement = metric.improvement
            
            # 应用指标权重
            weighted_improvement = improvement * metric.weight
            
            if metric_type not in type_improvements:
                type_improvements[metric_type] = []
                type_weights[metric_type] = self.metric_weights.get(metric_type, 1.0)
                
            type_improvements[metric_type].append(weighted_improvement)
            
        # 计算每个类型的平均改进
        type_avg_improvements = {}
        for metric_type, improvements in type_improvements.items():
            type_avg_improvements[metric_type] = (
                statistics.mean(improvements) if improvements else 0.0
            )
            
        # 计算加权总体改进
        total_weight = sum(type_weights.values())
        overall_improvement = sum(
            type_avg_improvements[t] * type_weights[t]
            for t in type_avg_improvements
        ) / total_weight if total_weight > 0 else 0.0
        
        # 生成详细分析
        details = {
            'type_improvements': {
                t.value: type_avg_improvements[t]
                for t in type_avg_improvements
            },
            'individual_metrics': [
                {
                    'name': m.name,
                    'type': m.metric_type.value,
                    'improvement': m.improvement,
                    'is_improved': m.is_improved
                }
                for m in all_metrics
            ],
            'improved_count': sum(1 for m in all_metrics if m.is_improved),
            'total_count': len(all_metrics)
        }
        
        logger.info(f"Comprehensive evaluation: overall_improvement={overall_improvement:.4f}")
        return overall_improvement, details
        
    def validate_evolution(
        self,
        evolutionary_change: EvolutionaryChange,
        old_metrics: Dict[str, Any],
        new_metrics: Dict[str, Any]
    ) -> ValidationResult:
        """
        验证进化改进效果
        
        Args:
            evolutionary_change: 进化变更
            old_metrics: 旧指标
            new_metrics: 新指标
            
        Returns:
            ValidationResult: 验证结果
        """
        all_metrics = []
        
        # 1. 性能改进测量
        performance_metrics = self.measure_performance_gain(
            old_metrics.get('performance', {}),
            new_metrics.get('performance', {})
        )
        all_metrics.extend(performance_metrics)
        
        # 2. 知识扩展评估
        knowledge_metrics = self.assess_knowledge_growth(
            old_metrics.get('knowledge', {}),
            new_metrics.get('knowledge', {})
        )
        all_metrics.extend(knowledge_metrics)
        
        # 3. 推理增强评估
        reasoning_metrics = self.evaluate_reasoning_improvement(
            old_metrics.get('reasoning', {}),
            new_metrics.get('reasoning', {})
        )
        all_metrics.extend(reasoning_metrics)
        
        # 4. 效率改进计算
        efficiency_metrics = self.calculate_efficiency_improvement(
            old_metrics.get('efficiency', {}),
            new_metrics.get('efficiency', {})
        )
        all_metrics.extend(efficiency_metrics)
        
        # 5. 综合评估
        overall_improvement, details = self.comprehensive_evaluation(all_metrics)
        
        # 决定是否接受进化
        is_valid = overall_improvement > self.improvement_threshold
        
        # 生成推荐
        recommendation = self._generate_recommendation(
            overall_improvement,
            all_metrics,
            details
        )
        
        # 计算置信度
        confidence = self._calculate_confidence(all_metrics)
        
        result = ValidationResult(
            is_valid=is_valid,
            overall_improvement=overall_improvement,
            metrics=all_metrics,
            recommendation=recommendation,
            confidence=confidence,
            details=details
        )
        
        # 保存验证历史
        self.validation_history.append(result)
        
        # 添加到待处理变更
        self.pending_changes.append(evolutionary_change)
        
        logger.info(
            f"Evolution validation: is_valid={is_valid}, "
            f"improvement={overall_improvement:.4f}, "
            f"recommendation={recommendation}"
        )
        
        return result
        
    def commit_evolutionary_change(
        self,
        change: EvolutionaryChange
    ) -> bool:
        """
        提交进化变更
        
        Args:
            change: 进化变更
            
        Returns:
            bool: 是否成功提交
        """
        if change in self.pending_changes:
            self.pending_changes.remove(change)
            self.committed_changes.append(change)
            logger.info(f"Committed evolutionary change: {change.change_id}")
            return True
        return False
        
    def rollback_evolutionary_change(
        self,
        change: EvolutionaryChange
    ) -> bool:
        """
        回滚进化变更
        
        Args:
            change: 进化变更
            
        Returns:
            bool: 是否成功回滚
        """
        if change in self.pending_changes:
            self.pending_changes.remove(change)
            self.rolled_back_changes.append(change)
            logger.info(f"Rolled back evolutionary change: {change.change_id}")
            return True
        return False
        
    def process_validation_result(
        self,
        result: ValidationResult,
        change: EvolutionaryChange
    ) -> str:
        """
        处理验证结果
        
        Args:
            result: 验证结果
            change: 进化变更
            
        Returns:
            str: 处理结果 ('committed', 'rolled_back', 'pending')
        """
        if result.recommendation == 'commit':
            self.commit_evolutionary_change(change)
            return 'committed'
        elif result.recommendation == 'rollback':
            self.rollback_evolutionary_change(change)
            return 'rolled_back'
        else:
            # partial_commit 或其他情况，保持待处理状态
            return 'pending'
            
    def _generate_recommendation(
        self,
        overall_improvement: float,
        metrics: List[ValidationMetric],
        details: Dict[str, Any]
    ) -> str:
        """生成推荐决策"""
        # 统计改进和退步的指标数量
        improved_count = details.get('improved_count', 0)
        total_count = details.get('total_count', 1)
        improvement_ratio = improved_count / total_count if total_count > 0 else 0
        
        # 决策逻辑
        if overall_improvement > self.improvement_threshold * 2 and improvement_ratio > 0.7:
            return 'commit'
        elif overall_improvement < -self.improvement_threshold:
            return 'rollback'
        elif overall_improvement > self.improvement_threshold and improvement_ratio > 0.5:
            return 'commit'
        elif overall_improvement > 0 and improvement_ratio > 0.6:
            return 'partial_commit'
        else:
            return 'rollback'
            
    def _calculate_confidence(
        self,
        metrics: List[ValidationMetric]
    ) -> float:
        """计算验证置信度"""
        if not metrics:
            return 0.0
            
        # 基于指标一致性计算置信度
        improvements = [m.improvement for m in metrics]
        
        # 如果所有改进都是正的或都是负的，置信度高
        all_positive = all(i >= 0 for i in improvements)
        all_negative = all(i <= 0 for i in improvements)
        
        if all_positive or all_negative:
            confidence = 0.9
        else:
            # 计算变异系数
            if statistics.mean(improvements) != 0:
                cv = abs(statistics.stdev(improvements) / statistics.mean(improvements)) if len(improvements) > 1 else 0
                confidence = max(0.3, 1.0 - cv)
            else:
                confidence = 0.5
                
        return min(1.0, confidence)
        
    def get_validation_statistics(self) -> Dict[str, Any]:
        """获取验证统计"""
        if not self.validation_history:
            return {}
            
        improvements = [r.overall_improvement for r in self.validation_history]
        valid_count = sum(1 for r in self.validation_history if r.is_valid)
        
        return {
            'total_validations': len(self.validation_history),
            'valid_count': valid_count,
            'valid_ratio': valid_count / len(self.validation_history),
            'avg_improvement': statistics.mean(improvements),
            'max_improvement': max(improvements),
            'min_improvement': min(improvements),
            'pending_changes': len(self.pending_changes),
            'committed_changes': len(self.committed_changes),
            'rolled_back_changes': len(self.rolled_back_changes)
        }
        
    def export_validation_history(self) -> Dict[str, Any]:
        """导出验证历史"""
        return {
            'validation_history': [
                {
                    'is_valid': r.is_valid,
                    'overall_improvement': r.overall_improvement,
                    'recommendation': r.recommendation,
                    'confidence': r.confidence,
                    'validated_at': r.validated_at.isoformat(),
                    'improved_metrics_count': len([m for m in r.metrics if m.is_improved]),
                    'total_metrics_count': len(r.metrics)
                }
                for r in self.validation_history
            ],
            'statistics': self.get_validation_statistics()
        }
