"""
自我进化框架 (Self-Evolution Framework)
========================================

基于最新自进化AI研究，为大模型实现"自己提升自己"的完整进化闭环系统。

模块包含:
- evolution_trigger: 进化触发机制
- parameter_optimizer: 模型参数自我优化
- architecture_evolution: 神经网络架构自我重塑
- training_data_generator: 自我训练数据生成
- evolutionary_algorithm: 进化算法实现
- multi_round_reflection: 多轮反思进化
- tool_creator: 工具能力自我扩展
- evolution_director: 智能进化导向系统
- evolution_validator: 进化效果验证
- main_evolution: 完整进化工作流

进化飞轮效应:
环境挑战 → 性能差距识别 → 自我优化 → 能力提升 → 应对更复杂挑战 → ...
"""

from .evolution_trigger import EvolutionTrigger
from .parameter_optimizer import SelfParameterOptimizer
from .architecture_evolution import NeuralArchitectureEvolution
from .training_data_generator import SelfTrainingDataGenerator
from .evolutionary_algorithm import EvolutionaryAlgorithm
from .multi_round_reflection import MultiRoundReflection
from .tool_creator import SelfToolCreator
from .evolution_director import IntelligentEvolutionDirector
from .evolution_validator import EvolutionValidator
from .main_evolution import main_evolution_cycle

__all__ = [
    'EvolutionTrigger',
    'SelfParameterOptimizer', 
    'NeuralArchitectureEvolution',
    'SelfTrainingDataGenerator',
    'EvolutionaryAlgorithm',
    'MultiRoundReflection',
    'SelfToolCreator',
    'IntelligentEvolutionDirector',
    'EvolutionValidator',
    'main_evolution_cycle'
]

__version__ = '1.0.0'
