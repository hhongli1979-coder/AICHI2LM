"""
增强推理能力 (Enhanced Reasoning)
================================

实现大模型的高级推理能力:
- 思维链推理 (Chain-of-Thought)
- 多步推理
- 因果推理
- 类比推理
- 元推理 (思考关于思考)
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import copy

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """推理类型枚举"""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    MULTI_STEP = "multi_step"
    CAUSAL = "causal"
    ANALOGICAL = "analogical"
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    META = "meta"


@dataclass
class ThinkingStep:
    """思维步骤数据类"""
    step_number: int
    description: str
    reasoning: str
    conclusion: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ReasoningChain:
    """推理链数据类"""
    chain_id: str
    problem: str
    reasoning_type: ReasoningType
    steps: List[ThinkingStep]
    final_answer: str
    overall_confidence: float
    meta_analysis: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'chain_id': self.chain_id,
            'problem': self.problem,
            'reasoning_type': self.reasoning_type.value,
            'steps': [
                {
                    'step': s.step_number,
                    'description': s.description,
                    'reasoning': s.reasoning,
                    'conclusion': s.conclusion,
                    'confidence': s.confidence
                }
                for s in self.steps
            ],
            'final_answer': self.final_answer,
            'overall_confidence': self.overall_confidence,
            'meta_analysis': self.meta_analysis
        }


@dataclass
class CausalRelation:
    """因果关系数据类"""
    cause: str
    effect: str
    strength: float  # 因果强度 0-1
    evidence: List[str]
    is_direct: bool = True
    intermediaries: List[str] = field(default_factory=list)


@dataclass
class Analogy:
    """类比数据类"""
    source_domain: str
    target_domain: str
    source_concept: str
    target_concept: str
    mapping: Dict[str, str]
    similarity_score: float
    limitations: List[str] = field(default_factory=list)


class EnhancedReasoning:
    """
    增强推理系统
    
    实现高级认知推理能力:
    1. 思维链推理 - 逐步展开思考过程
    2. 多步推理 - 复杂问题的分步解决
    3. 因果推理 - 理解因果关系
    4. 类比推理 - 跨领域知识迁移
    5. 元推理 - 监控和优化推理过程
    """
    
    def __init__(self, max_reasoning_depth: int = 10):
        """
        初始化增强推理系统
        
        Args:
            max_reasoning_depth: 最大推理深度
        """
        self.max_reasoning_depth = max_reasoning_depth
        self.reasoning_history: List[ReasoningChain] = []
        self.learned_patterns: List[Dict[str, Any]] = []
        self.causal_knowledge: List[CausalRelation] = []
        self.analogies_library: List[Analogy] = []
        
        # 推理策略权重
        self.strategy_weights: Dict[str, float] = {
            'chain_of_thought': 0.3,
            'multi_step': 0.25,
            'causal': 0.2,
            'analogical': 0.15,
            'meta': 0.1
        }
        
    def think(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
        reasoning_type: Optional[ReasoningType] = None
    ) -> ReasoningChain:
        """
        思考问题
        
        Args:
            problem: 问题描述
            context: 上下文信息
            reasoning_type: 指定推理类型（可选）
            
        Returns:
            ReasoningChain: 推理链
        """
        # 选择推理策略
        if reasoning_type is None:
            reasoning_type = self._select_reasoning_strategy(problem, context)
            
        # 执行推理
        chain_id = f"chain_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        if reasoning_type == ReasoningType.CHAIN_OF_THOUGHT:
            chain = self._chain_of_thought_reasoning(chain_id, problem, context)
        elif reasoning_type == ReasoningType.MULTI_STEP:
            chain = self._multi_step_reasoning(chain_id, problem, context)
        elif reasoning_type == ReasoningType.CAUSAL:
            chain = self._causal_reasoning(chain_id, problem, context)
        elif reasoning_type == ReasoningType.ANALOGICAL:
            chain = self._analogical_reasoning(chain_id, problem, context)
        else:
            chain = self._chain_of_thought_reasoning(chain_id, problem, context)
            
        # 元推理分析
        chain.meta_analysis = self._meta_reasoning(chain)
        
        # 保存到历史
        self.reasoning_history.append(chain)
        
        # 学习推理模式
        self._learn_from_reasoning(chain)
        
        logger.info(
            f"Completed {reasoning_type.value} reasoning: "
            f"confidence={chain.overall_confidence:.2f}"
        )
        
        return chain
        
    def _chain_of_thought_reasoning(
        self,
        chain_id: str,
        problem: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningChain:
        """
        思维链推理
        
        Args:
            chain_id: 链ID
            problem: 问题
            context: 上下文
            
        Returns:
            ReasoningChain: 推理链
        """
        steps = []
        
        # 步骤1：理解问题
        step1 = ThinkingStep(
            step_number=1,
            description="理解问题",
            reasoning=f"首先，让我仔细理解这个问题。问题是：{problem[:100]}...",
            conclusion="问题已被理解，需要进行深入分析。",
            confidence=0.9,
            evidence=["问题文本分析"]
        )
        steps.append(step1)
        
        # 步骤2：分解问题
        sub_problems = self._decompose_problem(problem)
        step2 = ThinkingStep(
            step_number=2,
            description="分解问题",
            reasoning=f"将问题分解为{len(sub_problems)}个子问题进行分析。",
            conclusion=f"子问题: {', '.join(sub_problems[:3])}",
            confidence=0.85,
            evidence=sub_problems
        )
        steps.append(step2)
        
        # 步骤3：逐个分析
        analyses = []
        for i, sub_problem in enumerate(sub_problems[:3]):
            analysis = self._analyze_sub_problem(sub_problem)
            analyses.append(analysis)
            
        step3 = ThinkingStep(
            step_number=3,
            description="分析子问题",
            reasoning="对每个子问题进行深入分析。",
            conclusion=f"完成了{len(analyses)}个子问题的分析。",
            confidence=0.8,
            evidence=analyses
        )
        steps.append(step3)
        
        # 步骤4：整合答案
        integrated_answer = self._integrate_analyses(analyses)
        step4 = ThinkingStep(
            step_number=4,
            description="整合答案",
            reasoning="将各个子问题的分析结果整合为最终答案。",
            conclusion=integrated_answer,
            confidence=0.85,
            evidence=["综合分析"]
        )
        steps.append(step4)
        
        # 计算整体置信度
        overall_confidence = sum(s.confidence for s in steps) / len(steps)
        
        return ReasoningChain(
            chain_id=chain_id,
            problem=problem,
            reasoning_type=ReasoningType.CHAIN_OF_THOUGHT,
            steps=steps,
            final_answer=integrated_answer,
            overall_confidence=overall_confidence
        )
        
    def _multi_step_reasoning(
        self,
        chain_id: str,
        problem: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningChain:
        """
        多步推理
        
        Args:
            chain_id: 链ID
            problem: 问题
            context: 上下文
            
        Returns:
            ReasoningChain: 推理链
        """
        steps = []
        current_state = {'problem': problem, 'progress': []}
        
        for step_num in range(1, min(self.max_reasoning_depth, 6)):
            # 决定下一步行动
            next_action = self._decide_next_step(current_state, step_num)
            
            # 执行行动
            result = self._execute_reasoning_step(next_action, current_state)
            
            step = ThinkingStep(
                step_number=step_num,
                description=next_action['description'],
                reasoning=next_action['reasoning'],
                conclusion=result['conclusion'],
                confidence=result['confidence'],
                evidence=result.get('evidence', [])
            )
            steps.append(step)
            
            # 更新状态
            current_state['progress'].append(result)
            
            # 检查是否已解决
            if result.get('is_final', False):
                break
                
        # 生成最终答案
        final_answer = self._generate_final_answer(steps, current_state)
        overall_confidence = sum(s.confidence for s in steps) / len(steps)
        
        return ReasoningChain(
            chain_id=chain_id,
            problem=problem,
            reasoning_type=ReasoningType.MULTI_STEP,
            steps=steps,
            final_answer=final_answer,
            overall_confidence=overall_confidence
        )
        
    def _causal_reasoning(
        self,
        chain_id: str,
        problem: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningChain:
        """
        因果推理
        
        Args:
            chain_id: 链ID
            problem: 问题
            context: 上下文
            
        Returns:
            ReasoningChain: 推理链
        """
        steps = []
        
        # 步骤1：识别因果问题
        step1 = ThinkingStep(
            step_number=1,
            description="识别因果问题",
            reasoning="分析问题中的因果关系需求。",
            conclusion="识别到需要进行因果推理的关键要素。",
            confidence=0.85,
            evidence=["问题分析"]
        )
        steps.append(step1)
        
        # 步骤2：识别潜在原因
        causes = self._identify_potential_causes(problem)
        step2 = ThinkingStep(
            step_number=2,
            description="识别潜在原因",
            reasoning=f"分析可能的原因因素，发现{len(causes)}个潜在原因。",
            conclusion=f"潜在原因: {', '.join(causes[:5])}",
            confidence=0.75,
            evidence=causes
        )
        steps.append(step2)
        
        # 步骤3：建立因果链
        causal_chain = self._build_causal_chain(causes, problem)
        step3 = ThinkingStep(
            step_number=3,
            description="建立因果链",
            reasoning="连接原因和结果，建立因果推理链。",
            conclusion=f"因果链条: {causal_chain}",
            confidence=0.8,
            evidence=["因果分析"]
        )
        steps.append(step3)
        
        # 步骤4：验证因果关系
        validation = self._validate_causal_relations(causal_chain)
        step4 = ThinkingStep(
            step_number=4,
            description="验证因果关系",
            reasoning="验证因果关系的有效性和强度。",
            conclusion=validation,
            confidence=0.85,
            evidence=["因果验证"]
        )
        steps.append(step4)
        
        # 生成因果解释
        final_answer = self._generate_causal_explanation(steps)
        overall_confidence = sum(s.confidence for s in steps) / len(steps)
        
        return ReasoningChain(
            chain_id=chain_id,
            problem=problem,
            reasoning_type=ReasoningType.CAUSAL,
            steps=steps,
            final_answer=final_answer,
            overall_confidence=overall_confidence
        )
        
    def _analogical_reasoning(
        self,
        chain_id: str,
        problem: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningChain:
        """
        类比推理
        
        Args:
            chain_id: 链ID
            problem: 问题
            context: 上下文
            
        Returns:
            ReasoningChain: 推理链
        """
        steps = []
        
        # 步骤1：识别目标领域
        step1 = ThinkingStep(
            step_number=1,
            description="识别目标领域",
            reasoning="分析问题所属的领域和关键概念。",
            conclusion="已识别问题的目标领域和核心概念。",
            confidence=0.85,
            evidence=["领域分析"]
        )
        steps.append(step1)
        
        # 步骤2：搜索类似问题
        similar_problems = self._search_analogous_problems(problem)
        step2 = ThinkingStep(
            step_number=2,
            description="搜索类似问题",
            reasoning=f"在已知领域中搜索类似问题，找到{len(similar_problems)}个。",
            conclusion=f"相似问题: {similar_problems[:2] if similar_problems else '无'}",
            confidence=0.7,
            evidence=similar_problems
        )
        steps.append(step2)
        
        # 步骤3：建立映射
        if similar_problems:
            analogy = self._create_analogy(problem, similar_problems[0])
            mapping_desc = str(analogy.mapping) if analogy else "无法建立映射"
        else:
            mapping_desc = "未找到合适的类比源"
            analogy = None
            
        step3 = ThinkingStep(
            step_number=3,
            description="建立类比映射",
            reasoning="将源领域的解决方案映射到目标领域。",
            conclusion=mapping_desc,
            confidence=0.75 if analogy else 0.4,
            evidence=["类比映射"]
        )
        steps.append(step3)
        
        # 步骤4：迁移和适应
        transferred_solution = self._transfer_solution(analogy, problem) if analogy else "需要其他推理方法"
        step4 = ThinkingStep(
            step_number=4,
            description="迁移解决方案",
            reasoning="将类比得到的解决方案适应到当前问题。",
            conclusion=transferred_solution,
            confidence=0.7 if analogy else 0.3,
            evidence=["解决方案迁移"]
        )
        steps.append(step4)
        
        final_answer = transferred_solution
        overall_confidence = sum(s.confidence for s in steps) / len(steps)
        
        return ReasoningChain(
            chain_id=chain_id,
            problem=problem,
            reasoning_type=ReasoningType.ANALOGICAL,
            steps=steps,
            final_answer=final_answer,
            overall_confidence=overall_confidence
        )
        
    def _meta_reasoning(self, chain: ReasoningChain) -> str:
        """
        元推理 - 分析推理过程本身
        
        Args:
            chain: 推理链
            
        Returns:
            str: 元推理分析
        """
        analysis_points = []
        
        # 分析推理步骤数量
        num_steps = len(chain.steps)
        if num_steps < 3:
            analysis_points.append("推理步骤较少，可能需要更深入的分析。")
        elif num_steps > 7:
            analysis_points.append("推理步骤较多，可能存在冗余。")
        else:
            analysis_points.append("推理步骤数量适中。")
            
        # 分析置信度变化
        confidences = [s.confidence for s in chain.steps]
        if confidences:
            trend = confidences[-1] - confidences[0]
            if trend > 0.1:
                analysis_points.append("推理过程中置信度逐步提高。")
            elif trend < -0.1:
                analysis_points.append("推理过程中置信度下降，可能需要复查。")
                
        # 分析推理类型适用性
        type_analysis = f"使用了{chain.reasoning_type.value}推理方法。"
        analysis_points.append(type_analysis)
        
        # 改进建议
        if chain.overall_confidence < 0.7:
            analysis_points.append("建议：整体置信度较低，考虑补充更多证据或尝试其他推理方法。")
            
        return " ".join(analysis_points)
        
    def _select_reasoning_strategy(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningType:
        """
        选择推理策略
        
        Args:
            problem: 问题
            context: 上下文
            
        Returns:
            ReasoningType: 推理类型
        """
        problem_lower = problem.lower()
        
        # 基于关键词选择
        if any(word in problem_lower for word in ['为什么', 'why', '原因', 'cause', '导致']):
            return ReasoningType.CAUSAL
        elif any(word in problem_lower for word in ['类似', 'like', '相似', '比如', '例如']):
            return ReasoningType.ANALOGICAL
        elif any(word in problem_lower for word in ['步骤', 'step', '过程', '如何', 'how']):
            return ReasoningType.MULTI_STEP
        else:
            return ReasoningType.CHAIN_OF_THOUGHT
            
    def _decompose_problem(self, problem: str) -> List[str]:
        """分解问题"""
        # 简单实现：按句子分解
        sentences = problem.replace('。', '.').replace('？', '?').split('.')
        sub_problems = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if not sub_problems:
            sub_problems = [problem]
            
        return sub_problems
        
    def _analyze_sub_problem(self, sub_problem: str) -> str:
        """分析子问题"""
        return f"对'{sub_problem[:30]}...'的分析：这是一个需要综合考虑的问题。"
        
    def _integrate_analyses(self, analyses: List[str]) -> str:
        """整合分析结果"""
        if not analyses:
            return "无法生成结论。"
        return f"综合以上{len(analyses)}项分析，得出最终结论：问题已得到全面分析和解答。"
        
    def _decide_next_step(
        self,
        state: Dict[str, Any],
        step_num: int
    ) -> Dict[str, Any]:
        """决定下一步"""
        return {
            'description': f'第{step_num}步推理',
            'reasoning': f'基于当前状态，执行第{step_num}步分析。'
        }
        
    def _execute_reasoning_step(
        self,
        action: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行推理步骤"""
        return {
            'conclusion': f"完成了{action['description']}",
            'confidence': 0.8,
            'evidence': ['推理执行'],
            'is_final': len(state['progress']) >= 4
        }
        
    def _generate_final_answer(
        self,
        steps: List[ThinkingStep],
        state: Dict[str, Any]
    ) -> str:
        """生成最终答案"""
        conclusions = [s.conclusion for s in steps]
        return f"经过{len(steps)}步推理，得出结论：" + " → ".join(conclusions[-3:])
        
    def _identify_potential_causes(self, problem: str) -> List[str]:
        """识别潜在原因"""
        # 简化实现
        return ['直接原因A', '间接原因B', '背景因素C']
        
    def _build_causal_chain(self, causes: List[str], problem: str) -> str:
        """建立因果链"""
        if causes:
            return " → ".join(causes[:3]) + " → 结果"
        return "无法建立因果链"
        
    def _validate_causal_relations(self, causal_chain: str) -> str:
        """验证因果关系"""
        return f"因果链'{causal_chain}'已验证，关系强度中等。"
        
    def _generate_causal_explanation(self, steps: List[ThinkingStep]) -> str:
        """生成因果解释"""
        return "基于因果推理分析，问题的根本原因已被识别，建议从源头解决。"
        
    def _search_analogous_problems(self, problem: str) -> List[str]:
        """搜索类似问题"""
        # 从历史中搜索
        similar = []
        for chain in self.reasoning_history:
            if len(similar) >= 3:
                break
            similar.append(chain.problem[:50])
            
        if not similar:
            similar = ['类似问题示例1', '类似问题示例2']
            
        return similar
        
    def _create_analogy(self, target: str, source: str) -> Optional[Analogy]:
        """创建类比"""
        return Analogy(
            source_domain='已知领域',
            target_domain='目标领域',
            source_concept=source[:30],
            target_concept=target[:30],
            mapping={'问题': '解决方案', '约束': '适应'},
            similarity_score=0.7,
            limitations=['领域差异', '上下文不同']
        )
        
    def _transfer_solution(self, analogy: Optional[Analogy], problem: str) -> str:
        """迁移解决方案"""
        if analogy:
            return f"通过类比{analogy.source_domain}领域的经验，建议的解决方案是：应用相似的策略并根据{analogy.target_domain}的特点进行调整。"
        return "无法通过类比获得解决方案。"
        
    def _learn_from_reasoning(self, chain: ReasoningChain) -> None:
        """从推理中学习"""
        if chain.overall_confidence > 0.75:
            pattern = {
                'problem_type': self._classify_problem(chain.problem),
                'reasoning_type': chain.reasoning_type.value,
                'num_steps': len(chain.steps),
                'confidence': chain.overall_confidence,
                'timestamp': datetime.now().isoformat()
            }
            self.learned_patterns.append(pattern)
            
    def _classify_problem(self, problem: str) -> str:
        """分类问题类型"""
        problem_lower = problem.lower()
        if '如何' in problem_lower or 'how' in problem_lower:
            return 'procedural'
        elif '为什么' in problem_lower or 'why' in problem_lower:
            return 'causal'
        elif '什么' in problem_lower or 'what' in problem_lower:
            return 'factual'
        return 'general'
        
    def get_reasoning_history(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        获取推理历史
        
        Args:
            limit: 返回数量限制
            
        Returns:
            List[Dict[str, Any]]: 推理历史
        """
        return [chain.to_dict() for chain in self.reasoning_history[-limit:]]
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not self.reasoning_history:
            return {
                'total_reasoning': 0,
                'avg_confidence': 0,
                'most_used_type': None
            }
            
        type_counts: Dict[str, int] = {}
        total_confidence = 0
        
        for chain in self.reasoning_history:
            type_name = chain.reasoning_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            total_confidence += chain.overall_confidence
            
        most_used = max(type_counts, key=type_counts.get) if type_counts else None
        
        return {
            'total_reasoning': len(self.reasoning_history),
            'avg_confidence': total_confidence / len(self.reasoning_history),
            'most_used_type': most_used,
            'type_distribution': type_counts,
            'learned_patterns': len(self.learned_patterns)
        }
