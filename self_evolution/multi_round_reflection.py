"""
多轮反思进化 (Multi-Round Reflection)
=====================================

实现通过多轮反思实现进化:
- 生成初步解决方案
- 深度反思优化
- 元认知优化
- 从反思中学习进化
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import copy

logger = logging.getLogger(__name__)


@dataclass
class Solution:
    """解决方案数据类"""
    content: str
    quality_score: float
    round_number: int
    improvements: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ReflectionInsight:
    """反思洞察数据类"""
    round_number: int
    insight_type: str  # 'strength', 'weakness', 'improvement', 'metacognitive'
    description: str
    actionable_suggestion: str
    confidence: float  # 0.0 - 1.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class MetacognitiveAnalysis:
    """元认知分析数据类"""
    thinking_patterns: List[str]
    reasoning_quality: float
    blind_spots: List[str]
    improvement_areas: List[str]
    recommended_strategies: List[str]


class MultiRoundReflection:
    """
    多轮反思进化
    
    通过多轮反思实现能力进化:
    1. 第一轮：生成初步解决方案
    2. 第二轮：深度反思优化
    3. 第三轮：元认知优化
    4. 从反思过程中学习和进化
    """
    
    def __init__(self, max_rounds: int = 3):
        """
        初始化多轮反思系统
        
        Args:
            max_rounds: 最大反思轮数
        """
        self.max_rounds = max_rounds
        self.reflection_history: List[Dict[str, Any]] = []
        self.learned_patterns: List[Dict[str, Any]] = []
        self.improvement_strategies: List[str] = []
        
    def generate_initial_solution(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Solution:
        """
        生成初步解决方案
        
        Args:
            problem: 问题描述
            context: 上下文信息
            
        Returns:
            Solution: 初步解决方案
        """
        # 分析问题
        problem_analysis = self._analyze_problem(problem)
        
        # 生成解决方案框架
        solution_framework = self._generate_solution_framework(problem_analysis)
        
        # 填充解决方案内容
        solution_content = self._fill_solution_content(solution_framework, context)
        
        # 评估初步质量
        quality_score = self._evaluate_solution_quality(solution_content, problem)
        
        solution = Solution(
            content=solution_content,
            quality_score=quality_score,
            round_number=1
        )
        
        logger.info(f"Generated initial solution with quality score: {quality_score:.3f}")
        return solution
        
    def reflect_on_solution(
        self,
        solution: Solution,
        problem: str
    ) -> ReflectionInsight:
        """
        深度反思优化
        
        Args:
            solution: 当前解决方案
            problem: 问题描述
            
        Returns:
            ReflectionInsight: 反思洞察
        """
        insights = []
        
        # 分析优势
        strengths = self._identify_strengths(solution.content)
        for strength in strengths:
            insights.append(ReflectionInsight(
                round_number=solution.round_number + 1,
                insight_type='strength',
                description=strength,
                actionable_suggestion='保持并强化这一优势',
                confidence=0.8
            ))
            
        # 分析弱点
        weaknesses = self._identify_weaknesses(solution.content, problem)
        for weakness in weaknesses:
            insights.append(ReflectionInsight(
                round_number=solution.round_number + 1,
                insight_type='weakness',
                description=weakness['description'],
                actionable_suggestion=weakness['suggestion'],
                confidence=0.7
            ))
            
        # 识别改进机会
        improvements = self._identify_improvements(solution.content, problem)
        for improvement in improvements:
            insights.append(ReflectionInsight(
                round_number=solution.round_number + 1,
                insight_type='improvement',
                description=improvement['description'],
                actionable_suggestion=improvement['action'],
                confidence=improvement.get('confidence', 0.6)
            ))
            
        # 选择最重要的洞察
        if insights:
            best_insight = max(insights, key=lambda x: x.confidence)
            logger.info(f"Reflection completed: {best_insight.insight_type}")
            return best_insight
            
        # 如果没有特定洞察，返回通用改进建议
        return ReflectionInsight(
            round_number=solution.round_number + 1,
            insight_type='improvement',
            description='解决方案整体质量可以通过细化和具体化来提升',
            actionable_suggestion='增加更多具体细节和示例',
            confidence=0.5
        )
        
    def improve_based_on_insight(
        self,
        solution: Solution,
        insight: ReflectionInsight
    ) -> Solution:
        """
        基于洞察改进解决方案
        
        Args:
            solution: 当前解决方案
            insight: 反思洞察
            
        Returns:
            Solution: 改进后的解决方案
        """
        improved_content = solution.content
        improvements = list(solution.improvements)
        
        if insight.insight_type == 'weakness':
            # 修复弱点
            improved_content = self._fix_weakness(
                improved_content,
                insight.description,
                insight.actionable_suggestion
            )
            improvements.append(f"修复: {insight.description}")
            
        elif insight.insight_type == 'improvement':
            # 应用改进
            improved_content = self._apply_improvement(
                improved_content,
                insight.actionable_suggestion
            )
            improvements.append(f"改进: {insight.actionable_suggestion}")
            
        elif insight.insight_type == 'strength':
            # 强化优势
            improved_content = self._enhance_strength(
                improved_content,
                insight.description
            )
            improvements.append(f"强化: {insight.description}")
            
        # 评估改进后的质量
        quality_score = solution.quality_score + (insight.confidence * 0.1)
        quality_score = min(1.0, quality_score)
        
        improved_solution = Solution(
            content=improved_content,
            quality_score=quality_score,
            round_number=solution.round_number + 1,
            improvements=improvements
        )
        
        logger.info(
            f"Improved solution from round {solution.round_number} to {improved_solution.round_number}"
        )
        return improved_solution
        
    def metacognitive_reflection(
        self,
        solution: Solution,
        reflection_insights: List[ReflectionInsight]
    ) -> MetacognitiveAnalysis:
        """
        元认知反思
        
        Args:
            solution: 当前解决方案
            reflection_insights: 之前的反思洞察
            
        Returns:
            MetacognitiveAnalysis: 元认知分析结果
        """
        # 分析思维模式
        thinking_patterns = self._analyze_thinking_patterns(reflection_insights)
        
        # 评估推理质量
        reasoning_quality = self._evaluate_reasoning_quality(solution, reflection_insights)
        
        # 识别盲点
        blind_spots = self._identify_blind_spots(solution.content, reflection_insights)
        
        # 确定改进领域
        improvement_areas = self._determine_improvement_areas(reflection_insights)
        
        # 推荐策略
        recommended_strategies = self._recommend_strategies(
            thinking_patterns,
            blind_spots,
            improvement_areas
        )
        
        analysis = MetacognitiveAnalysis(
            thinking_patterns=thinking_patterns,
            reasoning_quality=reasoning_quality,
            blind_spots=blind_spots,
            improvement_areas=improvement_areas,
            recommended_strategies=recommended_strategies
        )
        
        logger.info(f"Metacognitive analysis completed: reasoning_quality={reasoning_quality:.3f}")
        return analysis
        
    def apply_metacognitive_improvements(
        self,
        solution: Solution,
        analysis: MetacognitiveAnalysis
    ) -> Solution:
        """
        应用元认知改进
        
        Args:
            solution: 当前解决方案
            analysis: 元认知分析
            
        Returns:
            Solution: 最终优化的解决方案
        """
        improved_content = solution.content
        improvements = list(solution.improvements)
        
        # 应用推荐策略
        for strategy in analysis.recommended_strategies:
            improved_content = self._apply_strategy(improved_content, strategy)
            improvements.append(f"策略: {strategy}")
            
        # 填补盲点
        for blind_spot in analysis.blind_spots:
            improved_content = self._address_blind_spot(improved_content, blind_spot)
            improvements.append(f"填补: {blind_spot}")
            
        # 计算最终质量分数
        quality_boost = analysis.reasoning_quality * 0.1
        for _ in analysis.recommended_strategies:
            quality_boost += 0.02
            
        quality_score = min(1.0, solution.quality_score + quality_boost)
        
        final_solution = Solution(
            content=improved_content,
            quality_score=quality_score,
            round_number=solution.round_number + 1,
            improvements=improvements
        )
        
        logger.info(f"Applied metacognitive improvements, final quality: {quality_score:.3f}")
        return final_solution
        
    def learn_from_reflection_process(
        self,
        reflection_insights: List[ReflectionInsight]
    ) -> None:
        """
        从反思过程中学习
        
        Args:
            reflection_insights: 反思洞察列表
        """
        # 提取有效模式
        effective_patterns = []
        
        for insight in reflection_insights:
            if insight.confidence > 0.7:
                pattern = {
                    'type': insight.insight_type,
                    'description': insight.description,
                    'action': insight.actionable_suggestion,
                    'learned_at': datetime.now().isoformat()
                }
                effective_patterns.append(pattern)
                
        # 更新学习模式
        self.learned_patterns.extend(effective_patterns)
        
        # 提取改进策略
        for insight in reflection_insights:
            if insight.insight_type == 'improvement' and insight.confidence > 0.6:
                if insight.actionable_suggestion not in self.improvement_strategies:
                    self.improvement_strategies.append(insight.actionable_suggestion)
                    
        logger.info(
            f"Learned {len(effective_patterns)} patterns, "
            f"total strategies: {len(self.improvement_strategies)}"
        )
        
    def evolve_through_reflection(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Solution:
        """
        通过多轮反思实现进化
        
        完整的反思进化流程
        
        Args:
            problem: 问题描述
            context: 上下文信息
            
        Returns:
            Solution: 最终优化的解决方案
        """
        solutions = []
        reflection_insights = []
        
        # 第一轮：生成初步解决方案
        round1_solution = self.generate_initial_solution(problem, context)
        solutions.append(round1_solution)
        
        # 第二轮：深度反思优化
        round2_insight = self.reflect_on_solution(round1_solution, problem)
        round2_solution = self.improve_based_on_insight(round1_solution, round2_insight)
        solutions.append(round2_solution)
        reflection_insights.append(round2_insight)
        
        # 第三轮：元认知优化
        round3_insight = self.metacognitive_reflection(round2_solution, reflection_insights)
        final_solution = self.apply_metacognitive_improvements(round2_solution, round3_insight)
        solutions.append(final_solution)
        
        # 从反思中学习进化
        self.learn_from_reflection_process(reflection_insights)
        
        # 记录反思历史
        self.reflection_history.append({
            'problem': problem,
            'solutions': [
                {
                    'round': s.round_number,
                    'quality': s.quality_score,
                    'improvements': s.improvements
                }
                for s in solutions
            ],
            'insights': [
                {
                    'type': i.insight_type,
                    'description': i.description,
                    'confidence': i.confidence
                }
                for i in reflection_insights
            ],
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(
            f"Reflection evolution completed: "
            f"initial quality={round1_solution.quality_score:.3f}, "
            f"final quality={final_solution.quality_score:.3f}"
        )
        
        return final_solution
        
    def _analyze_problem(self, problem: str) -> Dict[str, Any]:
        """分析问题"""
        return {
            'problem_text': problem,
            'length': len(problem),
            'complexity': self._estimate_complexity(problem),
            'domain': self._identify_domain(problem),
            'key_concepts': self._extract_key_concepts(problem)
        }
        
    def _estimate_complexity(self, problem: str) -> float:
        """估计问题复杂度"""
        # 基于问题长度和关键词的简单复杂度估计
        base_complexity = min(1.0, len(problem) / 500)
        
        complexity_indicators = ['复杂', '困难', '挑战', '多步', '综合', '分析']
        for indicator in complexity_indicators:
            if indicator in problem:
                base_complexity += 0.1
                
        return min(1.0, base_complexity)
        
    def _identify_domain(self, problem: str) -> str:
        """识别问题领域"""
        domain_keywords = {
            'mathematics': ['数学', '计算', '方程', '公式', '证明'],
            'coding': ['代码', '编程', '程序', '算法', '实现'],
            'reasoning': ['推理', '逻辑', '分析', '判断', '推断'],
            'knowledge': ['知识', '概念', '定义', '解释', '原理'],
            'creativity': ['创意', '设计', '创作', '想象', '创新'],
        }
        
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in problem:
                    return domain
                    
        return 'general'
        
    def _extract_key_concepts(self, problem: str) -> List[str]:
        """提取关键概念"""
        # 简单实现：提取较长的词作为关键概念
        words = problem.replace('，', ' ').replace('。', ' ').split()
        key_concepts = [w for w in words if len(w) >= 2]
        return key_concepts[:5]
        
    def _generate_solution_framework(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成解决方案框架"""
        return {
            'introduction': '问题分析与理解',
            'main_body': '解决方案主体',
            'conclusion': '总结与建议',
            'complexity_level': analysis.get('complexity', 0.5),
            'domain': analysis.get('domain', 'general')
        }
        
    def _fill_solution_content(
        self,
        framework: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """填充解决方案内容"""
        content = f"""
## 问题分析

{framework.get('introduction', '对问题进行了全面分析')}

## 解决方案

{framework.get('main_body', '提供详细的解决步骤')}

### 步骤1: 理解问题
首先，需要充分理解问题的本质和要求。

### 步骤2: 制定方案
基于问题分析，制定相应的解决方案。

### 步骤3: 执行验证
执行方案并验证结果的正确性。

## 总结

{framework.get('conclusion', '提供最终的总结和建议')}
"""
        return content.strip()
        
    def _evaluate_solution_quality(self, content: str, problem: str) -> float:
        """评估解决方案质量"""
        score = 0.5  # 基础分
        
        # 长度评估
        if len(content) > 200:
            score += 0.1
        if len(content) > 500:
            score += 0.1
            
        # 结构评估
        if '##' in content:
            score += 0.1
        if '步骤' in content:
            score += 0.05
        if '总结' in content:
            score += 0.05
            
        # 相关性评估
        problem_words = set(problem.split())
        content_words = set(content.split())
        overlap = len(problem_words & content_words)
        score += min(0.1, overlap * 0.02)
        
        return min(1.0, score)
        
    def _identify_strengths(self, content: str) -> List[str]:
        """识别优势"""
        strengths = []
        
        if '步骤' in content:
            strengths.append('解决方案具有清晰的步骤结构')
        if '分析' in content:
            strengths.append('包含问题分析部分')
        if '总结' in content:
            strengths.append('提供了清晰的总结')
        if len(content) > 500:
            strengths.append('内容充实详细')
            
        return strengths
        
    def _identify_weaknesses(
        self,
        content: str,
        problem: str
    ) -> List[Dict[str, str]]:
        """识别弱点"""
        weaknesses = []
        
        if len(content) < 200:
            weaknesses.append({
                'description': '解决方案内容过于简短',
                'suggestion': '增加更多细节和解释'
            })
            
        if '示例' not in content and '例如' not in content:
            weaknesses.append({
                'description': '缺少具体示例',
                'suggestion': '添加具体示例来说明解决方案'
            })
            
        if '验证' not in content:
            weaknesses.append({
                'description': '缺少验证步骤',
                'suggestion': '添加解决方案的验证和测试说明'
            })
            
        return weaknesses
        
    def _identify_improvements(
        self,
        content: str,
        problem: str
    ) -> List[Dict[str, Any]]:
        """识别改进机会"""
        improvements = []
        
        improvements.append({
            'description': '可以增加更多实际应用场景',
            'action': '添加实际应用案例',
            'confidence': 0.6
        })
        
        if '图' not in content and '表' not in content:
            improvements.append({
                'description': '可以添加图表辅助说明',
                'action': '考虑添加流程图或示意图',
                'confidence': 0.5
            })
            
        return improvements
        
    def _fix_weakness(
        self,
        content: str,
        weakness: str,
        suggestion: str
    ) -> str:
        """修复弱点"""
        # 在内容末尾添加改进部分
        fix_content = f"\n\n### 补充说明\n\n{suggestion}\n"
        return content + fix_content
        
    def _apply_improvement(self, content: str, improvement: str) -> str:
        """应用改进"""
        improvement_content = f"\n\n### 改进\n\n{improvement}\n"
        return content + improvement_content
        
    def _enhance_strength(self, content: str, strength: str) -> str:
        """强化优势"""
        # 在相关部分添加强化说明
        enhancement = f"\n\n[优势强化: {strength}]\n"
        return content + enhancement
        
    def _analyze_thinking_patterns(
        self,
        insights: List[ReflectionInsight]
    ) -> List[str]:
        """分析思维模式"""
        patterns = []
        
        insight_types = [i.insight_type for i in insights]
        
        if insight_types.count('weakness') > insight_types.count('strength'):
            patterns.append('倾向于发现问题和弱点')
        else:
            patterns.append('倾向于识别优势和机会')
            
        if any(i.confidence > 0.8 for i in insights):
            patterns.append('在某些领域有高度自信的判断')
            
        return patterns
        
    def _evaluate_reasoning_quality(
        self,
        solution: Solution,
        insights: List[ReflectionInsight]
    ) -> float:
        """评估推理质量"""
        base_quality = solution.quality_score
        
        # 基于洞察的平均置信度调整
        if insights:
            avg_confidence = sum(i.confidence for i in insights) / len(insights)
            quality = base_quality * 0.7 + avg_confidence * 0.3
        else:
            quality = base_quality
            
        return quality
        
    def _identify_blind_spots(
        self,
        content: str,
        insights: List[ReflectionInsight]
    ) -> List[str]:
        """识别盲点"""
        blind_spots = []
        
        # 检查常见的盲点
        if '风险' not in content and '问题' not in content:
            blind_spots.append('未考虑潜在风险')
            
        if '替代' not in content and '其他' not in content:
            blind_spots.append('未考虑替代方案')
            
        if '边界' not in content and '限制' not in content:
            blind_spots.append('未明确边界条件')
            
        return blind_spots
        
    def _determine_improvement_areas(
        self,
        insights: List[ReflectionInsight]
    ) -> List[str]:
        """确定改进领域"""
        areas = set()
        
        for insight in insights:
            if insight.insight_type == 'weakness':
                areas.add('弱点修复')
            elif insight.insight_type == 'improvement':
                areas.add('功能增强')
                
        return list(areas)
        
    def _recommend_strategies(
        self,
        patterns: List[str],
        blind_spots: List[str],
        improvement_areas: List[str]
    ) -> List[str]:
        """推荐改进策略"""
        strategies = []
        
        # 基于盲点推荐策略
        for blind_spot in blind_spots:
            if '风险' in blind_spot:
                strategies.append('增加风险评估部分')
            elif '替代' in blind_spot:
                strategies.append('考虑并比较替代方案')
            elif '边界' in blind_spot:
                strategies.append('明确定义边界条件和限制')
                
        # 使用已学习的策略
        for learned_strategy in self.improvement_strategies[:3]:
            if learned_strategy not in strategies:
                strategies.append(learned_strategy)
                
        return strategies
        
    def _apply_strategy(self, content: str, strategy: str) -> str:
        """应用策略"""
        strategy_content = f"\n\n### {strategy}\n\n[根据策略进行的改进内容]\n"
        return content + strategy_content
        
    def _address_blind_spot(self, content: str, blind_spot: str) -> str:
        """填补盲点"""
        blind_spot_content = f"\n\n### 补充: {blind_spot}\n\n[针对盲点的补充说明]\n"
        return content + blind_spot_content
        
    def export_learning(self) -> Dict[str, Any]:
        """导出学习内容"""
        return {
            'learned_patterns': self.learned_patterns,
            'improvement_strategies': self.improvement_strategies,
            'reflection_history_count': len(self.reflection_history)
        }
