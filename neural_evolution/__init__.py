# -*- coding: utf-8 -*-
"""
神经进化架构 - Neural Evolution Architecture
实现具有自我进化能力的超级智能系统

This module implements a neural evolution architecture for TeleChat,
enabling self-evolution capabilities including:
- Darwin Godel Machine architecture
- Four-dimensional self-evolution system
- Self-training mechanisms
- Evolution monitoring and safety constraints
"""

from .darwin_godel_machine import DarwinGodelMachine
from .evolutionary_memory import EvolutionaryMemory
from .tool_evolution import ToolEvolutionSystem
from .multi_round_thinking import MultiRoundThinking
from .self_reward import SelfRewardingSystem
from .evolution_monitor import EvolutionMonitor
from .evolution_laws import EvolutionaryLaws
from .multimodal_brain import UnifiedMultimodalBrain, VoiceNeuralNetwork

__version__ = "1.0.0"
__all__ = [
    "DarwinGodelMachine",
    "EvolutionaryMemory",
    "ToolEvolutionSystem",
    "MultiRoundThinking",
    "SelfRewardingSystem",
    "EvolutionMonitor",
    "EvolutionaryLaws",
    "UnifiedMultimodalBrain",
    "VoiceNeuralNetwork",
]
