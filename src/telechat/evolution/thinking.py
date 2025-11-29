"""
TeleChat 多轮思考模块
实现深度推理和自我反思能力
"""

from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

from ..utils.logging import get_logger

logger = get_logger("telechat.evolution.thinking")


@dataclass
class ThoughtStep:
    """思考步骤"""
    round: int
    content: str
    reasoning: str
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round": self.round,
            "content": self.content,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }


@dataclass
class Reflection:
    """反思记录"""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    missed_aspects: List[str] = field(default_factory=list)
    improvement_directions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "missed_aspects": self.missed_aspects,
            "improvement_directions": self.improvement_directions
        }


@dataclass
class ThinkingResult:
    """思考结果"""
    final_answer: str
    thought_steps: List[ThoughtStep] = field(default_factory=list)
    reflections: List[Reflection] = field(default_factory=list)
    total_rounds: int = 0
    final_confidence: float = 0.0
    metacognitive_insights: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "thought_steps": [s.to_dict() for s in self.thought_steps],
            "reflections": [r.to_dict() for r in self.reflections],
            "total_rounds": self.total_rounds,
            "final_confidence": self.final_confidence,
            "metacognitive_insights": self.metacognitive_insights
        }


class MultiRoundThinking:
    """多轮深度思考系统"""
    
    def __init__(
        self,
        thinking_rounds: int = 3,
        reflection_depth: int = 2,
        metacognition_enabled: bool = True,
        confidence_threshold: float = 0.85
    ):
        """
        初始化多轮思考系统
        
        Args:
            thinking_rounds: 思考轮次
            reflection_depth: 反思深度
            metacognition_enabled: 是否启用元认知
            confidence_threshold: 置信度阈值
        """
        self.thinking_rounds = thinking_rounds
        self.reflection_depth = reflection_depth
        self.metacognition_enabled = metacognition_enabled
        self.confidence_threshold = confidence_threshold
        
        # 思考策略偏好
        self.strategy_preferences: Dict[str, float] = {
            "analytical": 0.3,
            "creative": 0.3,
            "systematic": 0.4
        }
        
        logger.info(f"多轮思考系统初始化: rounds={thinking_rounds}, depth={reflection_depth}")
    
    def think(
        self,
        problem: str,
        generator: Callable[[str, Optional[str]], str],
        evaluator: Optional[Callable[[str, str], float]] = None
    ) -> ThinkingResult:
        """
        对问题进行多轮思考
        
        Args:
            problem: 问题描述
            generator: 文本生成函数 (prompt, context) -> response
            evaluator: 评估函数 (problem, solution) -> score
            
        Returns:
            思考结果
        """
        logger.info(f"开始多轮思考，问题长度: {len(problem)}")
        
        thought_steps: List[ThoughtStep] = []
        reflections: List[Reflection] = []
        
        # 第一轮：初始思考
        round1_answer = self._initial_thinking(problem, generator)
        thought_steps.append(ThoughtStep(
            round=1,
            content=round1_answer,
            reasoning="初始分析和直觉响应",
            confidence=0.5
        ))
        
        # 后续轮次：反思和改进
        current_answer = round1_answer
        for round_num in range(2, self.thinking_rounds + 1):
            # 反思
            reflection = self._deep_reflect(problem, current_answer, thought_steps)
            reflections.append(reflection)
            
            # 基于反思改进
            improved_answer = self._improved_thinking(
                problem, current_answer, reflection, generator
            )
            
            # 评估置信度
            if evaluator:
                confidence = evaluator(problem, improved_answer)
            else:
                confidence = self._estimate_confidence(improved_answer, reflection)
            
            thought_steps.append(ThoughtStep(
                round=round_num,
                content=improved_answer,
                reasoning=f"基于反思进行改进: {', '.join(reflection.improvement_directions[:2])}",
                confidence=confidence
            ))
            
            current_answer = improved_answer
            
            # 如果置信度足够高，提前结束
            if confidence >= self.confidence_threshold:
                logger.info(f"第{round_num}轮达到置信度阈值，提前结束")
                break
        
        # 元认知反思
        metacognitive_insights = []
        if self.metacognition_enabled:
            metacognitive_insights = self._metacognitive_learning(thought_steps, reflections)
        
        # 选择最佳答案
        final_answer, final_confidence = self._select_best_answer(thought_steps)
        
        result = ThinkingResult(
            final_answer=final_answer,
            thought_steps=thought_steps,
            reflections=reflections,
            total_rounds=len(thought_steps),
            final_confidence=final_confidence,
            metacognitive_insights=metacognitive_insights
        )
        
        logger.info(f"思考完成: {len(thought_steps)}轮, 置信度: {final_confidence:.2f}")
        return result
    
    def _initial_thinking(self, problem: str, generator: Callable) -> str:
        """初始思考"""
        prompt = f"""请仔细分析以下问题并给出你的初步思考：

问题：{problem}

请提供你的分析和初步答案："""
        
        try:
            return generator(prompt, None)
        except Exception as e:
            logger.error(f"初始思考失败: {str(e)}")
            return f"初步分析：{problem}"
    
    def _deep_reflect(
        self,
        problem: str,
        current_answer: str,
        previous_steps: List[ThoughtStep]
    ) -> Reflection:
        """深度反思"""
        reflection = Reflection()
        
        # 分析优点
        reflection.strengths = self._identify_strengths(current_answer)
        
        # 分析弱点
        reflection.weaknesses = self._identify_weaknesses(problem, current_answer)
        
        # 识别遗漏
        reflection.missed_aspects = self._find_missed_aspects(problem, current_answer)
        
        # 提出改进方向
        reflection.improvement_directions = self._suggest_improvements(
            problem, current_answer, reflection.weaknesses, reflection.missed_aspects
        )
        
        return reflection
    
    def _identify_strengths(self, answer: str) -> List[str]:
        """识别答案优点"""
        strengths = []
        
        if len(answer) > 100:
            strengths.append("回答较为详细")
        
        if "因为" in answer or "所以" in answer or "首先" in answer:
            strengths.append("包含逻辑推理")
        
        if "例如" in answer or "比如" in answer:
            strengths.append("包含具体示例")
        
        if not strengths:
            strengths.append("提供了基本回答")
        
        return strengths
    
    def _identify_weaknesses(self, problem: str, answer: str) -> List[str]:
        """识别答案弱点"""
        weaknesses = []
        
        if len(answer) < 50:
            weaknesses.append("回答过于简短")
        
        if "可能" in answer or "也许" in answer:
            weaknesses.append("存在不确定性表述")
        
        # 检查是否回答了问题的关键部分
        problem_keywords = set(problem.split())
        answer_keywords = set(answer.split())
        overlap = len(problem_keywords & answer_keywords)
        if overlap < len(problem_keywords) * 0.3:
            weaknesses.append("可能偏离问题主题")
        
        return weaknesses
    
    def _find_missed_aspects(self, problem: str, answer: str) -> List[str]:
        """识别遗漏的方面"""
        missed = []
        
        # 检查是否遗漏了问题中的关键词
        question_words = ["为什么", "如何", "什么", "哪些", "多少"]
        for qw in question_words:
            if qw in problem and qw not in answer:
                missed.append(f"可能未充分回答'{qw}'相关的问题")
        
        return missed
    
    def _suggest_improvements(
        self,
        problem: str,
        answer: str,
        weaknesses: List[str],
        missed: List[str]
    ) -> List[str]:
        """提出改进建议"""
        suggestions = []
        
        if "回答过于简短" in weaknesses:
            suggestions.append("增加更多细节和解释")
        
        if "存在不确定性表述" in weaknesses:
            suggestions.append("提供更确定的分析和结论")
        
        if "可能偏离问题主题" in weaknesses:
            suggestions.append("更紧密地围绕问题核心进行回答")
        
        for m in missed:
            suggestions.append(f"补充: {m}")
        
        if not suggestions:
            suggestions.append("进一步完善和优化表述")
        
        return suggestions
    
    def _improved_thinking(
        self,
        problem: str,
        current_answer: str,
        reflection: Reflection,
        generator: Callable
    ) -> str:
        """基于反思的改进思考"""
        prompt = f"""基于以下反思，请改进你的回答：

原问题：{problem}

当前回答：{current_answer}

反思：
- 优点：{', '.join(reflection.strengths)}
- 需要改进：{', '.join(reflection.weaknesses)}
- 改进方向：{', '.join(reflection.improvement_directions)}

请提供改进后的回答："""
        
        try:
            return generator(prompt, current_answer)
        except Exception as e:
            logger.error(f"改进思考失败: {str(e)}")
            return current_answer
    
    def _estimate_confidence(self, answer: str, reflection: Reflection) -> float:
        """估算置信度"""
        base_confidence = 0.5
        
        # 根据优点加分
        base_confidence += len(reflection.strengths) * 0.1
        
        # 根据弱点减分
        base_confidence -= len(reflection.weaknesses) * 0.05
        
        # 根据答案长度调整
        if 100 < len(answer) < 500:
            base_confidence += 0.1
        
        return max(0.1, min(0.95, base_confidence))
    
    def _select_best_answer(self, thought_steps: List[ThoughtStep]) -> tuple:
        """选择最佳答案"""
        if not thought_steps:
            return "", 0.0
        
        # 选择置信度最高的
        best_step = max(thought_steps, key=lambda s: s.confidence)
        return best_step.content, best_step.confidence
    
    def _metacognitive_learning(
        self,
        thought_steps: List[ThoughtStep],
        reflections: List[Reflection]
    ) -> List[str]:
        """元认知学习：从思考过程中学习"""
        insights = []
        
        # 分析思考策略的有效性
        confidence_trend = [s.confidence for s in thought_steps]
        if len(confidence_trend) > 1:
            if confidence_trend[-1] > confidence_trend[0]:
                insights.append("多轮思考有效提升了回答质量")
            else:
                insights.append("初始思考可能已足够，需调整策略")
        
        # 分析反思的有效性
        total_improvements = sum(len(r.improvement_directions) for r in reflections)
        if total_improvements > 0:
            insights.append(f"共识别了{total_improvements}个改进方向")
        
        # 更新策略偏好
        self._update_strategy_preferences(thought_steps, reflections)
        insights.append(f"当前策略偏好: {self.strategy_preferences}")
        
        return insights
    
    def _update_strategy_preferences(
        self,
        thought_steps: List[ThoughtStep],
        reflections: List[Reflection]
    ):
        """更新策略偏好"""
        # 根据结果调整偏好
        if thought_steps:
            final_confidence = thought_steps[-1].confidence
            
            if final_confidence > 0.8:
                # 成功的策略权重增加
                for key in self.strategy_preferences:
                    self.strategy_preferences[key] = min(1.0, self.strategy_preferences[key] + 0.01)
            else:
                # 尝试调整策略
                strategies = list(self.strategy_preferences.keys())
                selected = strategies[len(thought_steps) % len(strategies)]
                self.strategy_preferences[selected] = min(1.0, self.strategy_preferences[selected] + 0.05)


class SelfRewardingSystem:
    """自我奖励系统"""
    
    def __init__(self):
        self.reward_history: List[Dict[str, Any]] = []
        self.reward_scaling = 1.0
    
    def evaluate(
        self,
        task: str,
        solution: str,
        criteria: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        自我评估
        
        Args:
            task: 任务描述
            solution: 解决方案
            criteria: 评估标准权重
            
        Returns:
            评估结果
        """
        criteria = criteria or {
            "correctness": 0.35,
            "completeness": 0.25,
            "efficiency": 0.15,
            "creativity": 0.15,
            "clarity": 0.10
        }
        
        scores = {
            "correctness": self._evaluate_correctness(task, solution),
            "completeness": self._evaluate_completeness(task, solution),
            "efficiency": self._evaluate_efficiency(solution),
            "creativity": self._evaluate_creativity(solution),
            "clarity": self._evaluate_clarity(solution)
        }
        
        # 加权综合
        final_score = sum(scores[k] * criteria[k] for k in scores)
        
        result = {
            "final_score": final_score,
            "detailed_scores": scores,
            "criteria_weights": criteria,
            "reward": self._compute_reward(final_score),
            "timestamp": datetime.now().isoformat()
        }
        
        self.reward_history.append(result)
        return result
    
    def _evaluate_correctness(self, task: str, solution: str) -> float:
        """评估正确性"""
        # 简化实现
        if len(solution) > 0:
            return 0.7 + 0.3 * min(len(solution) / 500, 1.0)
        return 0.0
    
    def _evaluate_completeness(self, task: str, solution: str) -> float:
        """评估完整性"""
        # 检查解决方案是否涵盖任务关键点
        task_words = set(task.lower().split())
        solution_words = set(solution.lower().split())
        
        overlap = len(task_words & solution_words)
        return min(1.0, overlap / max(len(task_words), 1) + 0.3)
    
    def _evaluate_efficiency(self, solution: str) -> float:
        """评估效率"""
        # 适中长度最好
        optimal_length = 300
        current_length = len(solution)
        
        if current_length < 50:
            return 0.3
        elif current_length > 1000:
            return 0.6
        else:
            return 0.5 + 0.5 * (1 - abs(current_length - optimal_length) / optimal_length)
    
    def _evaluate_creativity(self, solution: str) -> float:
        """评估创造性"""
        creativity_indicators = ["新颖", "创新", "独特", "另一种", "或者", "也可以"]
        count = sum(1 for ind in creativity_indicators if ind in solution)
        return min(1.0, 0.4 + count * 0.15)
    
    def _evaluate_clarity(self, solution: str) -> float:
        """评估清晰度"""
        # 检查结构化表述
        structure_indicators = ["首先", "其次", "然后", "最后", "1.", "2.", "-", "•"]
        count = sum(1 for ind in structure_indicators if ind in solution)
        
        return min(1.0, 0.5 + count * 0.1)
    
    def _compute_reward(self, score: float) -> float:
        """计算奖励值"""
        # 非线性奖励函数
        if score >= 0.9:
            return 1.0 * self.reward_scaling
        elif score >= 0.7:
            return 0.7 * self.reward_scaling
        elif score >= 0.5:
            return 0.4 * self.reward_scaling
        else:
            return 0.1 * self.reward_scaling
    
    def get_reward_trend(self) -> List[float]:
        """获取奖励趋势"""
        return [r["reward"] for r in self.reward_history]
    
    def get_average_score(self) -> float:
        """获取平均分数"""
        if not self.reward_history:
            return 0.0
        return sum(r["final_score"] for r in self.reward_history) / len(self.reward_history)
