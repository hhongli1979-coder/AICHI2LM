"""
进化触发机制 (Evolution Trigger)
================================

检测是否需要进化的触发机制，包括:
- 性能阈值检测
- 知识缺口检测
- 新挑战接收
- 优化机会识别
"""

import logging
from typing import Set, List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    accuracy: float = 0.0
    response_time: float = 0.0
    knowledge_coverage: float = 0.0
    reasoning_depth: float = 0.0
    creativity_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class KnowledgeGap:
    """知识缺口数据类"""
    domain: str
    description: str
    severity: float  # 0.0 - 1.0
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class Challenge:
    """新挑战数据类"""
    challenge_type: str
    difficulty: float  # 0.0 - 1.0
    domain: str
    description: str
    received_at: datetime = field(default_factory=datetime.now)


class EvolutionTrigger:
    """
    进化触发机制
    
    负责检测是否需要触发进化，包括:
    1. 性能阈值检测 - 当性能低于阈值时触发
    2. 知识缺口检测 - 当发现知识盲区时触发
    3. 新挑战接收 - 当接收到新的复杂挑战时触发
    4. 优化机会识别 - 当发现可优化空间时触发
    """
    
    def __init__(self, performance_threshold: float = 0.85):
        """
        初始化进化触发器
        
        Args:
            performance_threshold: 性能阈值，低于此值触发进化
        """
        self.performance_threshold = performance_threshold
        self.knowledge_gaps: Set[str] = set()
        self.pending_challenges: List[Challenge] = []
        self.optimization_opportunities: List[Dict[str, Any]] = []
        self.current_metrics: Optional[PerformanceMetrics] = None
        self.evolution_history: List[Dict[str, Any]] = []
        
    def update_performance_metrics(self, metrics: PerformanceMetrics) -> None:
        """
        更新当前性能指标
        
        Args:
            metrics: 性能指标数据
        """
        self.current_metrics = metrics
        logger.info(f"Updated performance metrics: accuracy={metrics.accuracy:.3f}")
        
    def performance_below_threshold(self) -> bool:
        """
        检测性能是否低于阈值
        
        Returns:
            bool: 如果性能低于阈值返回True
        """
        if self.current_metrics is None:
            logger.warning("No performance metrics available")
            return False
            
        below_threshold = self.current_metrics.accuracy < self.performance_threshold
        if below_threshold:
            logger.info(
                f"Performance below threshold: "
                f"{self.current_metrics.accuracy:.3f} < {self.performance_threshold:.3f}"
            )
        return below_threshold
        
    def detect_knowledge_gaps(self) -> bool:
        """
        检测是否存在知识缺口
        
        Returns:
            bool: 如果存在知识缺口返回True
        """
        has_gaps = len(self.knowledge_gaps) > 0
        if has_gaps:
            logger.info(f"Detected {len(self.knowledge_gaps)} knowledge gaps")
        return has_gaps
        
    def add_knowledge_gap(self, gap: KnowledgeGap) -> None:
        """
        添加知识缺口
        
        Args:
            gap: 知识缺口数据
        """
        self.knowledge_gaps.add(gap.domain)
        logger.info(f"Added knowledge gap in domain: {gap.domain}")
        
    def receive_new_challenges(self) -> bool:
        """
        检测是否接收到新挑战
        
        Returns:
            bool: 如果有待处理的新挑战返回True
        """
        has_challenges = len(self.pending_challenges) > 0
        if has_challenges:
            logger.info(f"Received {len(self.pending_challenges)} new challenges")
        return has_challenges
        
    def add_challenge(self, challenge: Challenge) -> None:
        """
        添加新挑战
        
        Args:
            challenge: 挑战数据
        """
        self.pending_challenges.append(challenge)
        logger.info(f"Added new challenge: {challenge.challenge_type}")
        
    def identify_optimization_opportunities(self) -> bool:
        """
        识别优化机会
        
        Returns:
            bool: 如果存在优化机会返回True
        """
        # 分析当前指标，识别优化空间
        opportunities = []
        
        if self.current_metrics:
            if self.current_metrics.response_time > 1.0:  # 响应时间大于1秒
                opportunities.append({
                    'type': 'response_time',
                    'current': self.current_metrics.response_time,
                    'target': 1.0,
                    'description': '响应时间优化'
                })
                
            if self.current_metrics.reasoning_depth < 0.8:
                opportunities.append({
                    'type': 'reasoning_depth',
                    'current': self.current_metrics.reasoning_depth,
                    'target': 0.8,
                    'description': '推理深度优化'
                })
                
            if self.current_metrics.creativity_score < 0.7:
                opportunities.append({
                    'type': 'creativity',
                    'current': self.current_metrics.creativity_score,
                    'target': 0.7,
                    'description': '创造力优化'
                })
        
        self.optimization_opportunities = opportunities
        has_opportunities = len(opportunities) > 0
        if has_opportunities:
            logger.info(f"Identified {len(opportunities)} optimization opportunities")
        return has_opportunities
        
    def should_evolve(self) -> bool:
        """
        检测是否需要进化
        
        综合评估所有触发条件，决定是否需要启动进化流程
        
        Returns:
            bool: 如果需要进化返回True
        """
        triggers = [
            ('performance_below_threshold', self.performance_below_threshold()),
            ('knowledge_gaps_detected', self.detect_knowledge_gaps()),
            ('new_challenges_received', self.receive_new_challenges()),
            ('optimization_opportunities', self.identify_optimization_opportunities())
        ]
        
        triggered = [(name, result) for name, result in triggers if result]
        should_evolve = len(triggered) > 0
        
        if should_evolve:
            trigger_names = [name for name, _ in triggered]
            logger.info(f"Evolution triggered by: {trigger_names}")
            
            # 记录进化历史
            self.evolution_history.append({
                'timestamp': datetime.now().isoformat(),
                'triggers': trigger_names,
                'metrics': {
                    'accuracy': self.current_metrics.accuracy if self.current_metrics else None,
                    'knowledge_gaps': len(self.knowledge_gaps),
                    'pending_challenges': len(self.pending_challenges)
                }
            })
            
        return should_evolve
        
    def get_evolution_priority(self) -> List[str]:
        """
        获取进化优先级列表
        
        根据触发条件的严重程度，返回按优先级排序的进化方向
        
        Returns:
            List[str]: 按优先级排序的进化方向列表
        """
        priorities = []
        
        # 性能问题优先级最高
        if self.current_metrics and self.current_metrics.accuracy < 0.7:
            priorities.append('critical_performance_improvement')
        elif self.performance_below_threshold():
            priorities.append('performance_improvement')
            
        # 高难度挑战次之
        high_difficulty_challenges = [
            c for c in self.pending_challenges if c.difficulty > 0.8
        ]
        if high_difficulty_challenges:
            priorities.append('high_difficulty_challenge_adaptation')
            
        # 知识缺口填补
        if self.knowledge_gaps:
            priorities.append('knowledge_gap_filling')
            
        # 常规优化
        if self.optimization_opportunities:
            priorities.append('general_optimization')
            
        return priorities
        
    def clear_processed_triggers(self) -> None:
        """清除已处理的触发器"""
        self.knowledge_gaps.clear()
        self.pending_challenges.clear()
        self.optimization_opportunities.clear()
        logger.info("Cleared all processed triggers")
        
    def export_state(self) -> Dict[str, Any]:
        """
        导出触发器状态
        
        Returns:
            Dict: 触发器当前状态
        """
        return {
            'performance_threshold': self.performance_threshold,
            'knowledge_gaps': list(self.knowledge_gaps),
            'pending_challenges_count': len(self.pending_challenges),
            'optimization_opportunities_count': len(self.optimization_opportunities),
            'current_metrics': {
                'accuracy': self.current_metrics.accuracy if self.current_metrics else None,
                'response_time': self.current_metrics.response_time if self.current_metrics else None,
            },
            'evolution_history_count': len(self.evolution_history)
        }
        
    def import_state(self, state: Dict[str, Any]) -> None:
        """
        导入触发器状态
        
        Args:
            state: 触发器状态数据
        """
        self.performance_threshold = state.get('performance_threshold', 0.85)
        self.knowledge_gaps = set(state.get('knowledge_gaps', []))
        logger.info("Imported evolution trigger state")
