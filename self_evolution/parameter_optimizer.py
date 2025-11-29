"""
模型参数自我优化 (Self Parameter Optimizer)
==========================================

实现模型参数的自我诊断和优化:
- 自我诊断性能瓶颈
- 生成优化策略
- 执行参数调整
- 验证优化效果
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import copy

logger = logging.getLogger(__name__)


@dataclass
class ParameterBottleneck:
    """参数瓶颈数据类"""
    parameter_name: str
    bottleneck_type: str  # 'convergence', 'gradient', 'overfitting', 'underfitting'
    severity: float  # 0.0 - 1.0
    suggested_adjustment: str
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationStrategy:
    """优化策略数据类"""
    strategy_name: str
    target_parameters: List[str]
    adjustment_type: str  # 'learning_rate', 'weight_decay', 'gradient_clip', etc.
    adjustment_value: Any
    expected_improvement: float
    priority: int = 1


@dataclass
class ParameterState:
    """参数状态数据类"""
    parameters: Dict[str, Any]
    performance_score: float
    timestamp: datetime = field(default_factory=datetime.now)


class SelfParameterOptimizer:
    """
    模型参数自我优化器
    
    实现四维自主进化中的参数优化维度:
    1. 自我诊断性能瓶颈
    2. 生成优化策略
    3. 执行参数调整
    4. 验证优化效果
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        初始化参数优化器
        
        Args:
            model_config: 模型配置
        """
        self.model_config = model_config or {}
        self.parameter_history: List[ParameterState] = []
        self.optimization_log: List[Dict[str, Any]] = []
        self.current_bottlenecks: List[ParameterBottleneck] = []
        
        # 默认优化参数范围
        self.param_ranges = {
            'learning_rate': (1e-6, 1e-2),
            'weight_decay': (0.0, 0.1),
            'gradient_clip': (0.0, 5.0),
            'warmup_ratio': (0.0, 0.2),
            'dropout': (0.0, 0.5),
        }
        
    def analyze_performance_bottlenecks(
        self,
        training_metrics: Dict[str, Any],
        validation_metrics: Dict[str, Any]
    ) -> List[ParameterBottleneck]:
        """
        分析性能瓶颈
        
        Args:
            training_metrics: 训练指标
            validation_metrics: 验证指标
            
        Returns:
            List[ParameterBottleneck]: 检测到的瓶颈列表
        """
        bottlenecks = []
        
        # 检测过拟合
        train_loss = training_metrics.get('loss', 0)
        val_loss = validation_metrics.get('loss', 0)
        if val_loss > 0 and train_loss > 0:
            overfitting_ratio = val_loss / train_loss
            if overfitting_ratio > 1.5:
                bottlenecks.append(ParameterBottleneck(
                    parameter_name='weight_decay',
                    bottleneck_type='overfitting',
                    severity=min(1.0, (overfitting_ratio - 1.0) / 2.0),
                    suggested_adjustment='增加正则化强度'
                ))
                
        # 检测欠拟合
        train_accuracy = training_metrics.get('accuracy', 1.0)
        if train_accuracy < 0.7:
            bottlenecks.append(ParameterBottleneck(
                parameter_name='learning_rate',
                bottleneck_type='underfitting',
                severity=1.0 - train_accuracy,
                suggested_adjustment='调整学习率或增加模型容量'
            ))
            
        # 检测梯度问题
        gradient_norm = training_metrics.get('gradient_norm', 0)
        if gradient_norm > 10.0:
            bottlenecks.append(ParameterBottleneck(
                parameter_name='gradient_clip',
                bottleneck_type='gradient',
                severity=min(1.0, gradient_norm / 100.0),
                suggested_adjustment='启用或降低梯度裁剪阈值'
            ))
        elif gradient_norm < 0.001:
            bottlenecks.append(ParameterBottleneck(
                parameter_name='learning_rate',
                bottleneck_type='gradient',
                severity=0.8,
                suggested_adjustment='增加学习率或检查模型初始化'
            ))
            
        # 检测收敛问题
        loss_history = training_metrics.get('loss_history', [])
        if len(loss_history) >= 10:
            recent_losses = loss_history[-10:]
            loss_std = self._calculate_std(recent_losses)
            loss_mean = sum(recent_losses) / len(recent_losses)
            
            # 检测loss震荡
            if loss_std / max(loss_mean, 1e-8) > 0.3:
                bottlenecks.append(ParameterBottleneck(
                    parameter_name='learning_rate',
                    bottleneck_type='convergence',
                    severity=min(1.0, loss_std / loss_mean),
                    suggested_adjustment='降低学习率以稳定收敛'
                ))
                
        self.current_bottlenecks = bottlenecks
        logger.info(f"Detected {len(bottlenecks)} performance bottlenecks")
        return bottlenecks
        
    def generate_optimization_strategy(
        self,
        bottlenecks: List[ParameterBottleneck]
    ) -> List[OptimizationStrategy]:
        """
        生成优化策略
        
        Args:
            bottlenecks: 性能瓶颈列表
            
        Returns:
            List[OptimizationStrategy]: 优化策略列表
        """
        strategies = []
        
        for bottleneck in bottlenecks:
            if bottleneck.bottleneck_type == 'overfitting':
                strategies.append(OptimizationStrategy(
                    strategy_name='regularization_increase',
                    target_parameters=['weight_decay', 'dropout'],
                    adjustment_type='weight_decay',
                    adjustment_value=self._calculate_new_weight_decay(bottleneck.severity),
                    expected_improvement=bottleneck.severity * 0.5,
                    priority=2
                ))
                
            elif bottleneck.bottleneck_type == 'underfitting':
                strategies.append(OptimizationStrategy(
                    strategy_name='learning_capacity_increase',
                    target_parameters=['learning_rate'],
                    adjustment_type='learning_rate',
                    adjustment_value=self._calculate_new_learning_rate('increase'),
                    expected_improvement=bottleneck.severity * 0.4,
                    priority=1
                ))
                
            elif bottleneck.bottleneck_type == 'gradient':
                if 'clip' in bottleneck.suggested_adjustment.lower():
                    strategies.append(OptimizationStrategy(
                        strategy_name='gradient_stabilization',
                        target_parameters=['gradient_clip'],
                        adjustment_type='gradient_clip',
                        adjustment_value=1.0,
                        expected_improvement=bottleneck.severity * 0.6,
                        priority=1
                    ))
                    
            elif bottleneck.bottleneck_type == 'convergence':
                strategies.append(OptimizationStrategy(
                    strategy_name='learning_rate_decay',
                    target_parameters=['learning_rate'],
                    adjustment_type='learning_rate',
                    adjustment_value=self._calculate_new_learning_rate('decrease'),
                    expected_improvement=bottleneck.severity * 0.5,
                    priority=1
                ))
                
        # 按优先级排序
        strategies.sort(key=lambda x: x.priority)
        logger.info(f"Generated {len(strategies)} optimization strategies")
        return strategies
        
    def adjust_parameters(
        self,
        current_params: Dict[str, Any],
        strategy: OptimizationStrategy
    ) -> Dict[str, Any]:
        """
        执行参数调整
        
        Args:
            current_params: 当前参数
            strategy: 优化策略
            
        Returns:
            Dict[str, Any]: 调整后的参数
        """
        new_params = copy.deepcopy(current_params)
        
        for param_name in strategy.target_parameters:
            if param_name in new_params:
                old_value = new_params[param_name]
                
                if strategy.adjustment_type == param_name:
                    new_params[param_name] = strategy.adjustment_value
                else:
                    # 根据策略类型进行相应调整
                    if isinstance(old_value, (int, float)):
                        adjustment_factor = 1.0 + (strategy.expected_improvement * 0.1)
                        new_params[param_name] = old_value * adjustment_factor
                        
                # 确保在有效范围内
                if param_name in self.param_ranges:
                    min_val, max_val = self.param_ranges[param_name]
                    new_params[param_name] = max(min_val, min(max_val, new_params[param_name]))
                    
                logger.info(
                    f"Adjusted {param_name}: {old_value} -> {new_params[param_name]}"
                )
                
        # 记录参数调整
        self.optimization_log.append({
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy.strategy_name,
            'adjustments': {
                param: {'old': current_params.get(param), 'new': new_params.get(param)}
                for param in strategy.target_parameters
            }
        })
        
        return new_params
        
    def validate_improvement(
        self,
        old_metrics: Dict[str, Any],
        new_metrics: Dict[str, Any],
        threshold: float = 0.01
    ) -> Tuple[bool, float]:
        """
        验证优化效果
        
        Args:
            old_metrics: 优化前指标
            new_metrics: 优化后指标
            threshold: 改进阈值
            
        Returns:
            Tuple[bool, float]: (是否有改进, 改进幅度)
        """
        old_score = self._calculate_overall_score(old_metrics)
        new_score = self._calculate_overall_score(new_metrics)
        
        improvement = new_score - old_score
        is_improved = improvement > threshold
        
        logger.info(
            f"Validation result: improvement={improvement:.4f}, "
            f"threshold={threshold}, passed={is_improved}"
        )
        
        return is_improved, improvement
        
    def evolve_parameters(
        self,
        current_params: Dict[str, Any],
        training_metrics: Dict[str, Any],
        validation_metrics: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], bool]:
        """
        执行完整的参数进化流程
        
        Args:
            current_params: 当前参数
            training_metrics: 训练指标
            validation_metrics: 验证指标
            
        Returns:
            Tuple[Dict[str, Any], bool]: (新参数, 是否有改进)
        """
        # 1. 自我诊断性能瓶颈
        bottlenecks = self.analyze_performance_bottlenecks(
            training_metrics, validation_metrics
        )
        
        if not bottlenecks:
            logger.info("No bottlenecks detected, parameters are optimal")
            return current_params, False
            
        # 2. 生成优化策略
        strategies = self.generate_optimization_strategy(bottlenecks)
        
        if not strategies:
            logger.info("No optimization strategies generated")
            return current_params, False
            
        # 3. 执行参数调整 (使用最高优先级策略)
        new_params = self.adjust_parameters(current_params, strategies[0])
        
        # 保存参数状态
        self.parameter_history.append(ParameterState(
            parameters=current_params,
            performance_score=self._calculate_overall_score(validation_metrics)
        ))
        
        return new_params, True
        
    def rollback_parameters(self) -> Optional[Dict[str, Any]]:
        """
        回滚到上一个参数状态
        
        Returns:
            Optional[Dict[str, Any]]: 上一个参数状态，如果不存在返回None
        """
        if len(self.parameter_history) > 0:
            last_state = self.parameter_history.pop()
            logger.info("Rolled back to previous parameter state")
            return last_state.parameters
        return None
        
    def get_best_parameters(self) -> Optional[Dict[str, Any]]:
        """
        获取历史最佳参数
        
        Returns:
            Optional[Dict[str, Any]]: 历史最佳参数
        """
        if not self.parameter_history:
            return None
            
        best_state = max(self.parameter_history, key=lambda x: x.performance_score)
        return best_state.parameters
        
    def _calculate_std(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        try:
            import statistics
            return statistics.pstdev(values)
        except Exception:
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            return variance ** 0.5
        
    def _calculate_new_weight_decay(self, severity: float) -> float:
        """计算新的权重衰减值"""
        current = self.model_config.get('weight_decay', 0.01)
        # 根据严重程度增加权重衰减
        new_value = current * (1 + severity)
        return min(0.1, new_value)  # 限制最大值
        
    def _calculate_new_learning_rate(self, direction: str) -> float:
        """计算新的学习率"""
        current = self.model_config.get('learning_rate', 3e-5)
        if direction == 'increase':
            return current * 1.5
        else:  # decrease
            return current * 0.5
            
    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """计算综合评分"""
        accuracy = metrics.get('accuracy', 0.0)
        loss = metrics.get('loss', float('inf'))
        
        # 综合评分: 高精度 + 低损失
        loss_score = 1.0 / (1.0 + loss)
        return accuracy * 0.7 + loss_score * 0.3
        
    def export_optimization_history(self) -> Dict[str, Any]:
        """导出优化历史"""
        return {
            'parameter_history': [
                {
                    'parameters': state.parameters,
                    'performance_score': state.performance_score,
                    'timestamp': state.timestamp.isoformat()
                }
                for state in self.parameter_history
            ],
            'optimization_log': self.optimization_log,
            'current_bottlenecks': [
                {
                    'parameter_name': b.parameter_name,
                    'bottleneck_type': b.bottleneck_type,
                    'severity': b.severity
                }
                for b in self.current_bottlenecks
            ]
        }
