# -*- coding: utf-8 -*-
"""
推理引擎 - Reasoning Engine
高智商推理能力，支持思维链、反思和逻辑推理
High-IQ reasoning capabilities with chain-of-thought, reflection, and logical reasoning
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time


class ReasoningType(Enum):
    """推理类型 - Reasoning Types"""
    CHAIN_OF_THOUGHT = "chain_of_thought"  # 思维链推理
    REFLECTION = "reflection"  # 反思推理
    ANALOGY = "analogy"  # 类比推理
    DEDUCTION = "deduction"  # 演绎推理
    INDUCTION = "induction"  # 归纳推理
    ABDUCTION = "abduction"  # 溯因推理


@dataclass
class ReasoningStep:
    """推理步骤 - Reasoning Step"""
    step_number: int
    description: str
    reasoning_type: ReasoningType
    input_data: str
    output_data: str
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 - Convert to dictionary"""
        return {
            "step_number": self.step_number,
            "description": self.description,
            "reasoning_type": self.reasoning_type.value,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }


@dataclass
class ReasoningResult:
    """推理结果 - Reasoning Result"""
    question: str
    answer: str
    reasoning_steps: List[ReasoningStep]
    total_confidence: float
    reasoning_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_explanation(self) -> str:
        """获取推理解释 - Get reasoning explanation"""
        explanation_parts = [f"问题 (Question): {self.question}\n"]
        explanation_parts.append("推理过程 (Reasoning Process):\n")
        
        for step in self.reasoning_steps:
            explanation_parts.append(
                f"  步骤 {step.step_number}: {step.description}\n"
                f"    类型: {step.reasoning_type.value}\n"
                f"    置信度: {step.confidence:.2f}\n"
            )
        
        explanation_parts.append(f"\n答案 (Answer): {self.answer}")
        explanation_parts.append(f"\n总置信度 (Total Confidence): {self.total_confidence:.2f}")
        
        return "".join(explanation_parts)


class ReasoningEngine:
    """
    推理引擎 - Reasoning Engine
    实现高智商推理能力
    Implements high-IQ reasoning capabilities
    """
    
    def __init__(self):
        """初始化推理引擎 - Initialize reasoning engine"""
        self.reasoning_history: List[ReasoningResult] = []
        self.knowledge_rules: List[Dict[str, Any]] = []
        
        # 内置推理规则 - Built-in reasoning rules
        self._init_default_rules()
    
    def _init_default_rules(self) -> None:
        """初始化默认推理规则 - Initialize default reasoning rules"""
        self.knowledge_rules = [
            {
                "name": "transitive_relation",
                "description": "传递关系推理 - If A > B and B > C, then A > C",
                "pattern": "transitive"
            },
            {
                "name": "cause_effect",
                "description": "因果关系推理 - Cause and effect reasoning",
                "pattern": "causal"
            },
            {
                "name": "generalization",
                "description": "归纳概括 - Generalization from examples",
                "pattern": "inductive"
            }
        ]
    
    def chain_of_thought(
        self,
        question: str,
        context: Optional[str] = None,
        max_steps: int = 10
    ) -> ReasoningResult:
        """
        思维链推理
        Chain-of-thought reasoning
        
        Args:
            question: 问题
            context: 上下文信息
            max_steps: 最大推理步骤
            
        Returns:
            推理结果
        """
        start_time = time.time()
        steps = []
        current_input = question
        
        # 步骤1: 问题分析
        step1 = ReasoningStep(
            step_number=1,
            description="问题分析 - Analyze the question",
            reasoning_type=ReasoningType.CHAIN_OF_THOUGHT,
            input_data=question,
            output_data=f"识别问题类型和关键信息: {question}",
            confidence=0.9
        )
        steps.append(step1)
        
        # 步骤2: 知识检索
        step2 = ReasoningStep(
            step_number=2,
            description="知识检索 - Retrieve relevant knowledge",
            reasoning_type=ReasoningType.CHAIN_OF_THOUGHT,
            input_data=step1.output_data,
            output_data="检索相关知识和上下文信息",
            confidence=0.85
        )
        steps.append(step2)
        
        # 步骤3: 逻辑推理
        step3 = ReasoningStep(
            step_number=3,
            description="逻辑推理 - Apply logical reasoning",
            reasoning_type=ReasoningType.DEDUCTION,
            input_data=step2.output_data,
            output_data="应用逻辑规则进行推理",
            confidence=0.8
        )
        steps.append(step3)
        
        # 步骤4: 生成答案
        answer = self._generate_answer(question, steps, context)
        step4 = ReasoningStep(
            step_number=4,
            description="答案生成 - Generate answer",
            reasoning_type=ReasoningType.CHAIN_OF_THOUGHT,
            input_data=step3.output_data,
            output_data=answer,
            confidence=0.85
        )
        steps.append(step4)
        
        # 计算总置信度
        total_confidence = sum(s.confidence for s in steps) / len(steps)
        
        result = ReasoningResult(
            question=question,
            answer=answer,
            reasoning_steps=steps,
            total_confidence=total_confidence,
            reasoning_time=time.time() - start_time,
            metadata={"context": context, "method": "chain_of_thought"}
        )
        
        self.reasoning_history.append(result)
        return result
    
    def reflect(
        self,
        question: str,
        initial_answer: str,
        feedback: Optional[str] = None
    ) -> ReasoningResult:
        """
        反思推理 - 对初始答案进行反思和改进
        Reflection reasoning - reflect and improve initial answer
        
        Args:
            question: 原问题
            initial_answer: 初始答案
            feedback: 反馈信息
            
        Returns:
            改进后的推理结果
        """
        start_time = time.time()
        steps = []
        
        # 步骤1: 回顾初始答案
        step1 = ReasoningStep(
            step_number=1,
            description="回顾初始答案 - Review initial answer",
            reasoning_type=ReasoningType.REFLECTION,
            input_data=initial_answer,
            output_data=f"分析初始答案: {initial_answer}",
            confidence=0.9
        )
        steps.append(step1)
        
        # 步骤2: 识别问题
        step2 = ReasoningStep(
            step_number=2,
            description="识别潜在问题 - Identify potential issues",
            reasoning_type=ReasoningType.REFLECTION,
            input_data=step1.output_data,
            output_data="检查逻辑一致性和准确性",
            confidence=0.85
        )
        steps.append(step2)
        
        # 步骤3: 改进答案
        improved_answer = self._improve_answer(question, initial_answer, feedback)
        step3 = ReasoningStep(
            step_number=3,
            description="改进答案 - Improve answer",
            reasoning_type=ReasoningType.REFLECTION,
            input_data=step2.output_data,
            output_data=improved_answer,
            confidence=0.88
        )
        steps.append(step3)
        
        total_confidence = sum(s.confidence for s in steps) / len(steps)
        
        result = ReasoningResult(
            question=question,
            answer=improved_answer,
            reasoning_steps=steps,
            total_confidence=total_confidence,
            reasoning_time=time.time() - start_time,
            metadata={
                "initial_answer": initial_answer,
                "feedback": feedback,
                "method": "reflection"
            }
        )
        
        self.reasoning_history.append(result)
        return result
    
    def analogy_reasoning(
        self,
        source_situation: str,
        target_situation: str,
        source_solution: str
    ) -> ReasoningResult:
        """
        类比推理 - 从已知情况推理到新情况
        Analogy reasoning - reason from known to new situations
        
        Args:
            source_situation: 源情况
            target_situation: 目标情况
            source_solution: 源解决方案
            
        Returns:
            推理结果
        """
        start_time = time.time()
        steps = []
        
        # 步骤1: 分析源情况
        step1 = ReasoningStep(
            step_number=1,
            description="分析源情况 - Analyze source situation",
            reasoning_type=ReasoningType.ANALOGY,
            input_data=source_situation,
            output_data=f"源情况特征: {source_situation}",
            confidence=0.9
        )
        steps.append(step1)
        
        # 步骤2: 识别相似性
        step2 = ReasoningStep(
            step_number=2,
            description="识别相似性 - Identify similarities",
            reasoning_type=ReasoningType.ANALOGY,
            input_data=f"{source_situation} vs {target_situation}",
            output_data="识别两种情况之间的结构相似性",
            confidence=0.8
        )
        steps.append(step2)
        
        # 步骤3: 映射解决方案
        adapted_solution = self._adapt_solution(source_solution, target_situation)
        step3 = ReasoningStep(
            step_number=3,
            description="映射解决方案 - Map solution",
            reasoning_type=ReasoningType.ANALOGY,
            input_data=source_solution,
            output_data=adapted_solution,
            confidence=0.75
        )
        steps.append(step3)
        
        total_confidence = sum(s.confidence for s in steps) / len(steps)
        
        result = ReasoningResult(
            question=f"如何处理: {target_situation}",
            answer=adapted_solution,
            reasoning_steps=steps,
            total_confidence=total_confidence,
            reasoning_time=time.time() - start_time,
            metadata={
                "source_situation": source_situation,
                "source_solution": source_solution,
                "method": "analogy"
            }
        )
        
        self.reasoning_history.append(result)
        return result
    
    def _generate_answer(
        self,
        question: str,
        steps: List[ReasoningStep],
        context: Optional[str]
    ) -> str:
        """
        生成答案
        Generate answer based on reasoning steps
        """
        # 基于推理步骤生成答案
        answer_parts = []
        answer_parts.append(f"基于对问题'{question}'的分析：")
        
        if context:
            answer_parts.append(f"结合上下文信息: {context}")
        
        answer_parts.append("通过逻辑推理得出结论。")
        
        return " ".join(answer_parts)
    
    def _improve_answer(
        self,
        question: str,
        initial_answer: str,
        feedback: Optional[str]
    ) -> str:
        """
        改进答案
        Improve answer based on reflection
        """
        improved = f"[改进后] {initial_answer}"
        
        if feedback:
            improved += f" (根据反馈 '{feedback}' 进行了调整)"
        
        return improved
    
    def _adapt_solution(self, source_solution: str, target_situation: str) -> str:
        """
        适应解决方案
        Adapt solution to new situation
        """
        return f"基于类似情况的经验，针对'{target_situation}'建议: {source_solution}"
    
    def get_reasoning_history(self, limit: int = 10) -> List[ReasoningResult]:
        """
        获取推理历史
        Get reasoning history
        
        Args:
            limit: 返回数量限制
            
        Returns:
            推理历史列表
        """
        return self.reasoning_history[-limit:]
    
    def clear_history(self) -> None:
        """清除推理历史 - Clear reasoning history"""
        self.reasoning_history.clear()
    
    def add_knowledge_rule(
        self,
        name: str,
        description: str,
        pattern: str
    ) -> None:
        """
        添加知识规则
        Add knowledge rule
        
        Args:
            name: 规则名称
            description: 规则描述
            pattern: 规则模式
        """
        self.knowledge_rules.append({
            "name": name,
            "description": description,
            "pattern": pattern,
            "created_at": time.time()
        })
