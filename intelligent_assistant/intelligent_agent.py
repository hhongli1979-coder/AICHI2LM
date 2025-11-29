# -*- coding: utf-8 -*-
"""
智能代理 - Intelligent Agent
整合记忆管理和推理引擎的高智商AI助手
Integrates memory management and reasoning engine for high-IQ AI assistant
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .memory_manager import MemoryManager
from .reasoning_engine import ReasoningEngine, ReasoningResult


class IntelligentAgent:
    """
    智能代理 - Intelligent Agent
    具备高智商、记忆功能和语言理解能力的AI助手
    AI assistant with high-IQ, memory capabilities, and language understanding
    """
    
    def __init__(
        self,
        name: str = "AICHI",
        personality: str = "helpful",
        memory_path: Optional[str] = None,
        short_term_capacity: int = 100,
        long_term_capacity: int = 1000
    ):
        """
        初始化智能代理
        Initialize intelligent agent
        
        Args:
            name: 代理名称
            personality: 个性特征
            memory_path: 记忆持久化路径
            short_term_capacity: 短期记忆容量
            long_term_capacity: 长期记忆容量
        """
        self.name = name
        self.personality = personality
        self.created_at = time.time()
        
        # 初始化记忆管理器
        self.memory = MemoryManager(
            short_term_capacity=short_term_capacity,
            long_term_capacity=long_term_capacity,
            persistence_path=memory_path
        )
        
        # 初始化推理引擎
        self.reasoning = ReasoningEngine()
        
        # 技能注册表
        self.skills: Dict[str, Callable] = {}
        
        # 对话状态
        self.conversation_state: Dict[str, Any] = {
            "current_topic": None,
            "emotion": "neutral",
            "engagement_level": 0.5
        }
        
        # 注册默认技能
        self._register_default_skills()
    
    def _register_default_skills(self) -> None:
        """注册默认技能 - Register default skills"""
        self.skills["greeting"] = self._skill_greeting
        self.skills["farewell"] = self._skill_farewell
        self.skills["help"] = self._skill_help
        self.skills["memory_summary"] = self._skill_memory_summary
        self.skills["reasoning"] = self._skill_reasoning
    
    def chat(
        self,
        user_input: str,
        use_reasoning: bool = True,
        save_memory: bool = True
    ) -> str:
        """
        对话接口 - 与用户进行智能对话
        Chat interface - intelligent conversation with user
        
        Args:
            user_input: 用户输入
            use_reasoning: 是否使用推理引擎
            save_memory: 是否保存到记忆
            
        Returns:
            AI回复
        """
        # 保存用户输入到记忆
        if save_memory:
            self.memory.add_conversation("user", user_input)
        
        # 分析用户意图
        intent = self._analyze_intent(user_input)
        
        # 获取相关上下文
        context = self._get_relevant_context(user_input)
        
        # 生成回复
        if use_reasoning and self._requires_reasoning(user_input):
            response = self._generate_reasoned_response(user_input, context)
        else:
            response = self._generate_response(user_input, intent, context)
        
        # 保存AI回复到记忆
        if save_memory:
            self.memory.add_conversation("bot", response)
        
        # 更新对话状态
        self._update_conversation_state(user_input, response)
        
        return response
    
    def _analyze_intent(self, user_input: str) -> str:
        """
        分析用户意图
        Analyze user intent
        
        Args:
            user_input: 用户输入
            
        Returns:
            识别的意图
        """
        input_lower = user_input.lower()
        
        # 简单的意图识别
        greeting_keywords = ["你好", "hello", "hi", "嗨", "早上好", "晚上好"]
        farewell_keywords = ["再见", "拜拜", "goodbye", "bye", "晚安"]
        help_keywords = ["帮助", "help", "怎么用", "使用方法"]
        question_keywords = ["为什么", "怎么", "什么", "如何", "是否", "?", "？"]
        
        for keyword in greeting_keywords:
            if keyword in input_lower:
                return "greeting"
        
        for keyword in farewell_keywords:
            if keyword in input_lower:
                return "farewell"
        
        for keyword in help_keywords:
            if keyword in input_lower:
                return "help"
        
        for keyword in question_keywords:
            if keyword in input_lower:
                return "question"
        
        return "general"
    
    def _requires_reasoning(self, user_input: str) -> bool:
        """
        判断是否需要推理
        Determine if reasoning is required
        """
        reasoning_indicators = [
            "为什么", "怎么", "如何", "分析", "推理", "思考",
            "解释", "原因", "why", "how", "explain", "reason"
        ]
        input_lower = user_input.lower()
        return any(indicator in input_lower for indicator in reasoning_indicators)
    
    def _get_relevant_context(self, user_input: str) -> str:
        """
        获取相关上下文
        Get relevant context
        
        Args:
            user_input: 用户输入
            
        Returns:
            相关上下文信息
        """
        # 获取最近对话
        recent_conversations = self.memory.get_recent_context(5)
        
        # 搜索相关记忆
        relevant_memories = self.memory.search_memory(user_input, limit=3)
        
        context_parts = []
        
        if recent_conversations:
            recent_text = "; ".join([
                f"{c['role']}: {c['content'][:50]}"
                for c in recent_conversations[-3:]
            ])
            context_parts.append(f"最近对话: {recent_text}")
        
        if relevant_memories:
            memory_text = "; ".join([
                m.content[:50] for m in relevant_memories
            ])
            context_parts.append(f"相关记忆: {memory_text}")
        
        return " | ".join(context_parts) if context_parts else ""
    
    def _generate_reasoned_response(self, user_input: str, context: str) -> str:
        """
        生成推理回复
        Generate reasoned response
        
        Args:
            user_input: 用户输入
            context: 上下文信息
            
        Returns:
            推理后的回复
        """
        # 使用思维链推理
        result = self.reasoning.chain_of_thought(user_input, context)
        
        response_parts = [
            f"让我来思考一下这个问题...\n",
            f"\n{result.answer}\n",
            f"\n(推理置信度: {result.total_confidence:.0%})"
        ]
        
        return "".join(response_parts)
    
    def _generate_response(
        self,
        user_input: str,
        intent: str,
        context: str
    ) -> str:
        """
        生成普通回复
        Generate normal response
        
        Args:
            user_input: 用户输入
            intent: 用户意图
            context: 上下文信息
            
        Returns:
            回复内容
        """
        # 检查是否有对应的技能
        if intent in self.skills:
            return self.skills[intent](user_input, context)
        
        # 生成通用回复
        return self._generate_general_response(user_input, context)
    
    def _generate_general_response(self, user_input: str, context: str) -> str:
        """
        生成通用回复
        Generate general response
        """
        # 基于个性特征生成回复
        if self.personality == "helpful":
            prefix = "我很乐意帮助你。"
        elif self.personality == "friendly":
            prefix = "嗨！"
        else:
            prefix = ""
        
        return f"{prefix}关于'{user_input}'，我理解你的问题。让我来帮你分析一下。"
    
    def _update_conversation_state(self, user_input: str, response: str) -> None:
        """
        更新对话状态
        Update conversation state
        """
        # 更新参与度
        input_length = len(user_input)
        if input_length > 50:
            self.conversation_state["engagement_level"] = min(
                1.0, self.conversation_state["engagement_level"] + 0.1
            )
        
        # 检测情感（简化版）
        positive_words = ["好", "棒", "喜欢", "谢谢", "感谢"]
        negative_words = ["不好", "糟糕", "讨厌", "烦"]
        
        input_lower = user_input.lower()
        if any(word in input_lower for word in positive_words):
            self.conversation_state["emotion"] = "positive"
        elif any(word in input_lower for word in negative_words):
            self.conversation_state["emotion"] = "negative"
        else:
            self.conversation_state["emotion"] = "neutral"
    
    # ==================== 技能实现 ====================
    
    def _skill_greeting(self, user_input: str, context: str) -> str:
        """问候技能 - Greeting skill"""
        greetings = [
            f"你好！我是{self.name}，一个具有记忆和推理能力的智能助手。有什么可以帮助你的吗？",
            f"嗨！很高兴见到你。我是{self.name}，随时准备为你服务！",
            f"你好呀！我是{self.name}。今天想聊些什么？"
        ]
        
        # 根据交互次数选择不同的问候
        interaction_count = self.memory.user_profile.get("interaction_count", 0)
        if interaction_count > 10:
            return f"欢迎回来！很高兴再次见到你。有什么新的问题要讨论吗？"
        
        return greetings[interaction_count % len(greetings)]
    
    def _skill_farewell(self, user_input: str, context: str) -> str:
        """告别技能 - Farewell skill"""
        # 保存记忆
        if self.memory.persistence_path:
            self.memory.save_to_file()
        
        return f"再见！和你聊天很愉快。我会记住我们的对话，下次见！"
    
    def _skill_help(self, user_input: str, context: str) -> str:
        """帮助技能 - Help skill"""
        help_text = f"""
{self.name} - 智能AI助手使用指南

🧠 核心能力：
1. 记忆功能 - 我能记住我们的对话历史和重要信息
2. 推理能力 - 我能进行逻辑推理和思维链分析
3. 语言理解 - 我能理解中英文，进行自然对话

💡 使用建议：
- 直接用自然语言和我交流
- 问复杂问题时，我会进行推理分析
- 你可以问我记住了什么（记忆摘要）

📊 当前状态：
- 记忆条目: {len(self.memory.short_term_memory)} 条短期, {len(self.memory.long_term_memory)} 条长期
- 对话次数: {self.memory.user_profile.get('interaction_count', 0)}

有任何问题，随时问我！
"""
        return help_text.strip()
    
    def _skill_memory_summary(self, user_input: str, context: str) -> str:
        """记忆摘要技能 - Memory summary skill"""
        summary = self.memory.get_memory_summary()
        
        return f"""
📊 记忆摘要：
- 短期记忆: {summary['short_term_count']} 条
- 长期记忆: {summary['long_term_count']} 条
- 知识库: {summary['knowledge_count']} 项
- 对话记录: {summary['conversation_count']} 条
- 总交互次数: {summary['user_profile']['interaction_count']}
"""
    
    def _skill_reasoning(self, user_input: str, context: str) -> str:
        """推理技能 - Reasoning skill"""
        result = self.reasoning.chain_of_thought(user_input, context)
        return result.get_explanation()
    
    # ==================== 高级功能 ====================
    
    def register_skill(self, name: str, handler: Callable) -> None:
        """
        注册新技能
        Register new skill
        
        Args:
            name: 技能名称
            handler: 技能处理函数
        """
        self.skills[name] = handler
    
    def learn_knowledge(self, key: str, value: Any) -> None:
        """
        学习新知识
        Learn new knowledge
        
        Args:
            key: 知识键
            value: 知识值
        """
        self.memory.add_knowledge(key, value)
        self.memory.add_memory(
            content=f"学习了新知识: {key} = {value}",
            memory_type="long_term",
            importance=8.0,
            metadata={"type": "knowledge", "key": key}
        )
    
    def recall_knowledge(self, key: str) -> Optional[Any]:
        """
        回忆知识
        Recall knowledge
        
        Args:
            key: 知识键
            
        Returns:
            知识值或None
        """
        return self.memory.get_knowledge(key)
    
    def reflect_on_conversation(self) -> str:
        """
        对话反思 - 反思最近的对话
        Reflect on recent conversation
        
        Returns:
            反思结果
        """
        recent = self.memory.get_recent_context(10)
        if not recent:
            return "还没有足够的对话历史进行反思。"
        
        # 分析对话模式
        user_messages = [c for c in recent if c.get("role") == "user"]
        bot_messages = [c for c in recent if c.get("role") == "bot"]
        
        reflection_parts = [
            "📝 对话反思:\n",
            f"- 最近{len(recent)}轮对话中，用户发送了{len(user_messages)}条消息\n",
            f"- 当前情感状态: {self.conversation_state['emotion']}\n",
            f"- 参与度水平: {self.conversation_state['engagement_level']:.0%}\n"
        ]
        
        return "".join(reflection_parts)
    
    def save_state(self) -> None:
        """保存状态 - Save state"""
        if self.memory.persistence_path:
            self.memory.save_to_file()
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取代理状态
        Get agent status
        
        Returns:
            状态信息
        """
        return {
            "name": self.name,
            "personality": self.personality,
            "created_at": self.created_at,
            "memory_summary": self.memory.get_memory_summary(),
            "conversation_state": self.conversation_state,
            "skills": list(self.skills.keys()),
            "reasoning_history_count": len(self.reasoning.reasoning_history)
        }
