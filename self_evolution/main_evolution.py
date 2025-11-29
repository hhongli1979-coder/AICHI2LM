"""
完整进化工作流 (Main Evolution Cycle)
=====================================

主进化循环实现:
1. 环境感知与需求分析
2. 制定进化策略
3. 执行进化操作
4. 验证进化效果
5. 学习进化经验
6. 准备下一轮进化

进化飞轮效应:
环境挑战 → 性能差距识别 → 自我优化 → 能力提升 → 应对更复杂挑战 → ...
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

from .evolution_trigger import EvolutionTrigger, PerformanceMetrics, Challenge
from .parameter_optimizer import SelfParameterOptimizer
from .architecture_evolution import NeuralArchitectureEvolution
from .training_data_generator import SelfTrainingDataGenerator
from .evolutionary_algorithm import EvolutionaryAlgorithm, EvolutionConfig, Agent
from .multi_round_reflection import MultiRoundReflection
from .tool_creator import SelfToolCreator, ToolCapability
from .evolution_director import IntelligentEvolutionDirector, EvolutionDimension
from .evolution_validator import EvolutionValidator, EvolutionaryChange

logger = logging.getLogger(__name__)


@dataclass
class EvolutionCycleResult:
    """进化循环结果数据类"""
    cycle_id: int
    success: bool
    improvements: Dict[str, float]
    changes_committed: int
    changes_rolled_back: int
    duration_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EvolutionStrategy:
    """进化策略数据类"""
    strategy_name: str
    target_dimensions: List[str]
    priority: int
    estimated_improvement: float
    required_resources: Dict[str, Any]
    risk_level: float


class SelfEvolutionSystem:
    """
    自我进化系统
    
    整合所有进化组件，实现完整的自我进化闭环:
    - 进化触发机制
    - 参数自我优化
    - 架构自我重塑
    - 训练数据自我生成
    - 进化算法
    - 多轮反思
    - 工具自我创建
    - 智能进化导向
    - 进化效果验证
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化自我进化系统
        
        Args:
            config: 系统配置
        """
        self.config = config or {}
        
        # 初始化所有进化组件
        self.trigger = EvolutionTrigger(
            performance_threshold=self.config.get('performance_threshold', 0.85)
        )
        self.parameter_optimizer = SelfParameterOptimizer()
        self.architecture_evolution = NeuralArchitectureEvolution()
        self.data_generator = SelfTrainingDataGenerator()
        self.evolutionary_algorithm = EvolutionaryAlgorithm(
            config=EvolutionConfig(
                population_size=self.config.get('population_size', 50),
                max_generations=self.config.get('max_generations', 100)
            )
        )
        self.reflection = MultiRoundReflection()
        self.tool_creator = SelfToolCreator()
        self.director = IntelligentEvolutionDirector()
        self.validator = EvolutionValidator(
            improvement_threshold=self.config.get('improvement_threshold', 0.02)
        )
        
        # 进化状态
        self.cycle_count = 0
        self.evolution_history: List[EvolutionCycleResult] = []
        self.current_state: Dict[str, Any] = {}
        self.learned_experiences: List[Dict[str, Any]] = []
        
    def analyze_environment_and_needs(
        self,
        current_metrics: Dict[str, Any],
        external_challenges: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        环境感知与需求分析
        
        Args:
            current_metrics: 当前性能指标
            external_challenges: 外部挑战
            
        Returns:
            Dict[str, Any]: 环境分析结果和进化需求
        """
        logger.info("Step 1: Analyzing environment and needs")
        
        # 更新性能指标
        performance = PerformanceMetrics(
            accuracy=current_metrics.get('accuracy', 0.5),
            response_time=current_metrics.get('response_time', 1.0),
            knowledge_coverage=current_metrics.get('knowledge_coverage', 0.5),
            reasoning_depth=current_metrics.get('reasoning_depth', 0.5),
            creativity_score=current_metrics.get('creativity_score', 0.5)
        )
        self.trigger.update_performance_metrics(performance)
        
        # 添加外部挑战
        if external_challenges:
            for challenge_data in external_challenges:
                challenge = Challenge(
                    challenge_type=challenge_data.get('type', 'unknown'),
                    difficulty=challenge_data.get('difficulty', 0.5),
                    domain=challenge_data.get('domain', 'general'),
                    description=challenge_data.get('description', '')
                )
                self.trigger.add_challenge(challenge)
                
        # 检测是否需要进化
        needs_evolution = self.trigger.should_evolve()
        evolution_priorities = self.trigger.get_evolution_priority()
        
        # 使用导向系统评估能力
        capability_assessments = self.director.assess_current_capabilities(current_metrics)
        
        analysis_result = {
            'needs_evolution': needs_evolution,
            'evolution_priorities': evolution_priorities,
            'capability_assessments': {
                dim.value: {
                    'score': assess.score,
                    'confidence': assess.confidence
                }
                for dim, assess in capability_assessments.items()
            },
            'trigger_state': self.trigger.export_state(),
            'timestamp': datetime.now().isoformat()
        }
        
        return analysis_result
        
    def formulate_evolution_strategy(
        self,
        evolution_needs: Dict[str, Any]
    ) -> List[EvolutionStrategy]:
        """
        制定进化策略
        
        Args:
            evolution_needs: 进化需求分析结果
            
        Returns:
            List[EvolutionStrategy]: 进化策略列表
        """
        logger.info("Step 2: Formulating evolution strategy")
        
        strategies = []
        priorities = evolution_needs.get('evolution_priorities', [])
        
        # 根据优先级制定策略
        for priority_type in priorities:
            if priority_type == 'critical_performance_improvement':
                strategies.append(EvolutionStrategy(
                    strategy_name='intensive_parameter_optimization',
                    target_dimensions=['accuracy', 'reasoning_depth'],
                    priority=1,
                    estimated_improvement=0.15,
                    required_resources={'compute': 'high', 'time': 'long'},
                    risk_level=0.3
                ))
                
            elif priority_type == 'performance_improvement':
                strategies.append(EvolutionStrategy(
                    strategy_name='standard_optimization',
                    target_dimensions=['accuracy', 'efficiency'],
                    priority=2,
                    estimated_improvement=0.08,
                    required_resources={'compute': 'medium', 'time': 'medium'},
                    risk_level=0.2
                ))
                
            elif priority_type == 'high_difficulty_challenge_adaptation':
                strategies.append(EvolutionStrategy(
                    strategy_name='capability_expansion',
                    target_dimensions=['knowledge_breadth', 'adaptability'],
                    priority=2,
                    estimated_improvement=0.1,
                    required_resources={'compute': 'medium', 'data': 'new'},
                    risk_level=0.4
                ))
                
            elif priority_type == 'knowledge_gap_filling':
                strategies.append(EvolutionStrategy(
                    strategy_name='knowledge_augmentation',
                    target_dimensions=['knowledge_breadth'],
                    priority=3,
                    estimated_improvement=0.12,
                    required_resources={'data': 'domain_specific', 'time': 'medium'},
                    risk_level=0.2
                ))
                
            elif priority_type == 'general_optimization':
                strategies.append(EvolutionStrategy(
                    strategy_name='holistic_improvement',
                    target_dimensions=['efficiency', 'robustness'],
                    priority=4,
                    estimated_improvement=0.05,
                    required_resources={'compute': 'low', 'time': 'short'},
                    risk_level=0.1
                ))
                
        # 按优先级排序
        strategies.sort(key=lambda x: x.priority)
        
        return strategies
        
    def execute_evolutionary_operations(
        self,
        strategies: List[EvolutionStrategy]
    ) -> List[EvolutionaryChange]:
        """
        执行进化操作
        
        Args:
            strategies: 进化策略列表
            
        Returns:
            List[EvolutionaryChange]: 进化变更列表
        """
        logger.info("Step 3: Executing evolutionary operations")
        
        changes = []
        
        for strategy in strategies[:3]:  # 每轮最多执行3个策略
            change_id = f"change_{self.cycle_count}_{len(changes)}"
            
            if strategy.strategy_name == 'intensive_parameter_optimization':
                # 执行参数优化
                old_state = {'parameters': self.current_state.get('parameters', {})}
                
                # 模拟参数优化
                new_params, improved = self.parameter_optimizer.evolve_parameters(
                    current_params=old_state['parameters'],
                    training_metrics={'accuracy': 0.7, 'loss': 0.5},
                    validation_metrics={'accuracy': 0.65, 'loss': 0.6}
                )
                
                changes.append(EvolutionaryChange(
                    change_id=change_id,
                    change_type='parameter',
                    description=f'Parameter optimization via {strategy.strategy_name}',
                    old_state=old_state,
                    new_state={'parameters': new_params}
                ))
                
            elif strategy.strategy_name == 'capability_expansion':
                # 执行架构进化
                old_arch = self.architecture_evolution.current_architecture
                
                new_arch, improved = self.architecture_evolution.self_redesign_architecture(
                    performance_metrics={'accuracy': 0.7},
                    resource_metrics={'inference_time_ms': 500, 'memory_usage_gb': 10}
                )
                
                changes.append(EvolutionaryChange(
                    change_id=change_id,
                    change_type='architecture',
                    description=f'Architecture evolution via {strategy.strategy_name}',
                    old_state={'architecture': old_arch},
                    new_state={'architecture': new_arch}
                ))
                
            elif strategy.strategy_name == 'knowledge_augmentation':
                # 生成训练数据
                training_data = self.data_generator.create_training_data(
                    num_samples=100,
                    domains=['knowledge', 'reasoning']
                )
                
                changes.append(EvolutionaryChange(
                    change_id=change_id,
                    change_type='knowledge',
                    description=f'Knowledge expansion via {strategy.strategy_name}',
                    old_state={'training_data_count': 0},
                    new_state={'training_data_count': len(training_data)}
                ))
                
            else:
                # 使用多轮反思进行通用优化
                problem = f"Optimize for {', '.join(strategy.target_dimensions)}"
                solution = self.reflection.evolve_through_reflection(problem)
                
                changes.append(EvolutionaryChange(
                    change_id=change_id,
                    change_type='general',
                    description=f'General optimization via {strategy.strategy_name}',
                    old_state={'solution_quality': 0.5},
                    new_state={'solution_quality': solution.quality_score}
                ))
                
        return changes
        
    def validate_evolutionary_changes(
        self,
        changes: List[EvolutionaryChange],
        old_metrics: Dict[str, Any],
        new_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        验证进化效果
        
        Args:
            changes: 进化变更列表
            old_metrics: 旧指标
            new_metrics: 新指标
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        logger.info("Step 4: Validating evolutionary changes")
        
        validation_results = []
        committed_count = 0
        rolled_back_count = 0
        
        for change in changes:
            # 验证每个变更
            result = self.validator.validate_evolution(
                evolutionary_change=change,
                old_metrics=old_metrics,
                new_metrics=new_metrics
            )
            
            # 处理验证结果
            action = self.validator.process_validation_result(result, change)
            
            if action == 'committed':
                committed_count += 1
            elif action == 'rolled_back':
                rolled_back_count += 1
                
            validation_results.append({
                'change_id': change.change_id,
                'is_valid': result.is_valid,
                'improvement': result.overall_improvement,
                'recommendation': result.recommendation,
                'action_taken': action
            })
            
        return {
            'results': validation_results,
            'committed_count': committed_count,
            'rolled_back_count': rolled_back_count,
            'overall_success_rate': committed_count / len(changes) if changes else 0
        }
        
    def learn_from_evolution_experience(
        self,
        validation_results: Dict[str, Any]
    ) -> None:
        """
        学习进化经验
        
        Args:
            validation_results: 验证结果
        """
        logger.info("Step 5: Learning from evolution experience")
        
        experience = {
            'cycle': self.cycle_count,
            'timestamp': datetime.now().isoformat(),
            'validation_results': validation_results,
            'lessons_learned': []
        }
        
        # 分析成功和失败的模式
        for result in validation_results.get('results', []):
            if result['is_valid']:
                experience['lessons_learned'].append({
                    'type': 'success',
                    'change_id': result['change_id'],
                    'improvement': result['improvement'],
                    'lesson': 'This type of evolution was effective'
                })
            else:
                experience['lessons_learned'].append({
                    'type': 'failure',
                    'change_id': result['change_id'],
                    'improvement': result['improvement'],
                    'lesson': 'This approach needs adjustment'
                })
                
        # 从反思系统学习
        reflection_learning = self.reflection.export_learning()
        experience['reflection_patterns'] = reflection_learning.get('learned_patterns', [])
        
        self.learned_experiences.append(experience)
        
        # 动态调整进化目标
        self.director.adjust_goals_dynamically({
            'success_rate': validation_results.get('overall_success_rate', 0)
        })
        
    def prepare_next_evolution_cycle(
        self,
        validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        准备下一轮进化
        
        Args:
            validation_results: 验证结果
            
        Returns:
            Dict[str, Any]: 下一轮准备状态
        """
        logger.info("Step 6: Preparing next evolution cycle")
        
        # 清除已处理的触发器
        self.trigger.clear_processed_triggers()
        
        # 更新当前状态
        self.current_state['last_evolution_cycle'] = self.cycle_count
        self.current_state['last_success_rate'] = validation_results.get('overall_success_rate', 0)
        
        # 获取进化导向状态
        evolution_status = self.director.get_evolution_status()
        
        # 确定下一轮重点
        next_focus = []
        for goal_name, goal_data in evolution_status.get('goals', {}).items():
            if goal_data.get('gap', 0) > 0.1:
                next_focus.append(goal_name)
                
        preparation = {
            'cycle_completed': self.cycle_count,
            'next_focus_areas': next_focus,
            'evolution_status': evolution_status,
            'ready_for_next_cycle': True,
            'timestamp': datetime.now().isoformat()
        }
        
        return preparation


def main_evolution_cycle(
    system: Optional[SelfEvolutionSystem] = None,
    current_metrics: Optional[Dict[str, Any]] = None,
    external_challenges: Optional[List[Dict[str, Any]]] = None
) -> EvolutionCycleResult:
    """
    主进化循环
    
    执行完整的自我进化流程:
    1. 环境感知与需求分析
    2. 制定进化策略
    3. 执行进化操作
    4. 验证进化效果
    5. 学习进化经验
    6. 准备下一轮进化
    
    Args:
        system: 自我进化系统实例
        current_metrics: 当前性能指标
        external_challenges: 外部挑战
        
    Returns:
        EvolutionCycleResult: 进化循环结果
    """
    start_time = datetime.now()
    
    # 初始化系统
    if system is None:
        system = SelfEvolutionSystem()
        
    if current_metrics is None:
        current_metrics = {
            'accuracy': 0.75,
            'response_time': 0.5,
            'knowledge_coverage': 0.6,
            'reasoning_depth': 0.7,
            'creativity_score': 0.65
        }
        
    # 增加循环计数
    system.cycle_count += 1
    cycle_id = system.cycle_count
    
    logger.info(f"Starting evolution cycle {cycle_id}")
    
    # 步骤1：环境感知与需求分析
    evolution_needs = system.analyze_environment_and_needs(
        current_metrics,
        external_challenges
    )
    
    # 如果不需要进化，提前返回
    if not evolution_needs['needs_evolution']:
        logger.info("No evolution needed at this time")
        return EvolutionCycleResult(
            cycle_id=cycle_id,
            success=True,
            improvements={},
            changes_committed=0,
            changes_rolled_back=0,
            duration_seconds=(datetime.now() - start_time).total_seconds()
        )
        
    # 步骤2：制定进化策略
    strategies = system.formulate_evolution_strategy(evolution_needs)
    
    if not strategies:
        logger.info("No strategies formulated")
        return EvolutionCycleResult(
            cycle_id=cycle_id,
            success=True,
            improvements={},
            changes_committed=0,
            changes_rolled_back=0,
            duration_seconds=(datetime.now() - start_time).total_seconds()
        )
        
    # 步骤3：执行进化操作
    changes = system.execute_evolutionary_operations(strategies)
    
    # 步骤4：验证进化效果
    # 模拟新指标（实际应该重新评估）
    new_metrics = {
        'accuracy': current_metrics.get('accuracy', 0) + 0.05,
        'response_time': current_metrics.get('response_time', 1) * 0.95,
        'knowledge_coverage': current_metrics.get('knowledge_coverage', 0) + 0.03,
        'reasoning_depth': current_metrics.get('reasoning_depth', 0) + 0.02,
        'creativity_score': current_metrics.get('creativity_score', 0) + 0.01
    }
    
    # 构建完整指标结构
    old_full_metrics = {
        'performance': {'accuracy': current_metrics.get('accuracy', 0), 'loss': 0.3},
        'knowledge': {'domains': ['general'], 'knowledge_depth': 0.5},
        'reasoning': {'logical_consistency': 0.6, 'multi_step_accuracy': 0.5},
        'efficiency': {'inference_speed': 100, 'memory_usage_gb': 10}
    }
    
    new_full_metrics = {
        'performance': {'accuracy': new_metrics.get('accuracy', 0), 'loss': 0.25},
        'knowledge': {'domains': ['general', 'science'], 'knowledge_depth': 0.55},
        'reasoning': {'logical_consistency': 0.65, 'multi_step_accuracy': 0.55},
        'efficiency': {'inference_speed': 110, 'memory_usage_gb': 9.5}
    }
    
    validation_results = system.validate_evolutionary_changes(
        changes,
        old_full_metrics,
        new_full_metrics
    )
    
    # 步骤5：学习进化经验
    system.learn_from_evolution_experience(validation_results)
    
    # 步骤6：准备下一轮进化
    preparation = system.prepare_next_evolution_cycle(validation_results)
    
    # 计算改进
    improvements = {
        'accuracy': new_metrics['accuracy'] - current_metrics['accuracy'],
        'response_time': current_metrics['response_time'] - new_metrics['response_time'],
        'knowledge_coverage': new_metrics['knowledge_coverage'] - current_metrics['knowledge_coverage'],
        'reasoning_depth': new_metrics['reasoning_depth'] - current_metrics['reasoning_depth'],
        'creativity_score': new_metrics['creativity_score'] - current_metrics['creativity_score']
    }
    
    # 创建结果
    duration = (datetime.now() - start_time).total_seconds()
    
    result = EvolutionCycleResult(
        cycle_id=cycle_id,
        success=validation_results['overall_success_rate'] > 0.5,
        improvements=improvements,
        changes_committed=validation_results['committed_count'],
        changes_rolled_back=validation_results['rolled_back_count'],
        duration_seconds=duration
    )
    
    # 保存到历史
    system.evolution_history.append(result)
    
    logger.info(
        f"Evolution cycle {cycle_id} completed: "
        f"success={result.success}, "
        f"committed={result.changes_committed}, "
        f"rolled_back={result.changes_rolled_back}"
    )
    
    return result


def run_continuous_evolution(
    num_cycles: int = 10,
    initial_metrics: Optional[Dict[str, Any]] = None
) -> List[EvolutionCycleResult]:
    """
    运行持续进化
    
    Args:
        num_cycles: 进化循环次数
        initial_metrics: 初始指标
        
    Returns:
        List[EvolutionCycleResult]: 所有循环的结果
    """
    system = SelfEvolutionSystem()
    results = []
    
    current_metrics = initial_metrics or {
        'accuracy': 0.70,
        'response_time': 1.0,
        'knowledge_coverage': 0.50,
        'reasoning_depth': 0.60,
        'creativity_score': 0.55
    }
    
    for i in range(num_cycles):
        logger.info(f"Starting continuous evolution cycle {i + 1}/{num_cycles}")
        
        # 执行进化循环
        result = main_evolution_cycle(
            system=system,
            current_metrics=current_metrics
        )
        results.append(result)
        
        # 更新指标（模拟进步）
        for key in current_metrics:
            improvement = result.improvements.get(key, 0)
            if key == 'response_time':
                current_metrics[key] = max(0.1, current_metrics[key] - abs(improvement))
            else:
                current_metrics[key] = min(1.0, current_metrics[key] + abs(improvement))
                
    # 输出最终统计
    successful_cycles = sum(1 for r in results if r.success)
    total_improvements = {
        key: sum(r.improvements.get(key, 0) for r in results)
        for key in current_metrics.keys()
    }
    
    logger.info(f"Continuous evolution completed: {successful_cycles}/{num_cycles} successful")
    logger.info(f"Total improvements: {total_improvements}")
    
    return results
