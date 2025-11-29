# -*- coding: utf-8 -*-
"""
记忆管理系统 - Memory Management System
支持短期记忆、长期记忆和知识库存储
Supports short-term memory, long-term memory, and knowledge storage
"""

import json
import os
import time
from typing import Any, Dict, List, Optional
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MemoryEntry:
    """记忆条目 - Memory Entry"""
    content: str
    timestamp: float = field(default_factory=time.time)
    importance: float = 1.0
    memory_type: str = "short_term"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 - Convert to dictionary"""
        return {
            "content": self.content,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "memory_type": self.memory_type,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """从字典创建 - Create from dictionary"""
        return cls(
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time()),
            importance=data.get("importance", 1.0),
            memory_type=data.get("memory_type", "short_term"),
            metadata=data.get("metadata", {})
        )


class MemoryManager:
    """
    记忆管理器 - Memory Manager
    管理对话历史、知识存储和上下文记忆
    Manages conversation history, knowledge storage, and context memory
    """
    
    def __init__(
        self,
        short_term_capacity: int = 100,
        long_term_capacity: int = 1000,
        persistence_path: Optional[str] = None
    ):
        """
        初始化记忆管理器
        Initialize memory manager
        
        Args:
            short_term_capacity: 短期记忆容量
            long_term_capacity: 长期记忆容量
            persistence_path: 持久化存储路径
        """
        self.short_term_capacity = short_term_capacity
        self.long_term_capacity = long_term_capacity
        self.persistence_path = persistence_path
        
        # 短期记忆 - Short-term memory (recent conversations)
        self.short_term_memory: deque = deque(maxlen=short_term_capacity)
        
        # 长期记忆 - Long-term memory (important information)
        self.long_term_memory: List[MemoryEntry] = []
        
        # 知识库 - Knowledge base
        self.knowledge_base: Dict[str, Any] = {}
        
        # 对话历史 - Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
        # 用户画像 - User profile
        self.user_profile: Dict[str, Any] = {
            "preferences": {},
            "interests": [],
            "interaction_count": 0,
            "first_interaction": None,
            "last_interaction": None
        }
        
        # 加载持久化数据
        if persistence_path and os.path.exists(persistence_path):
            self.load_from_file(persistence_path)
    
    def add_memory(
        self,
        content: str,
        memory_type: str = "short_term",
        importance: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryEntry:
        """
        添加记忆
        Add memory entry
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型 ("short_term" or "long_term")
            importance: 重要性分数 (0-10)
            metadata: 额外元数据
            
        Returns:
            创建的记忆条目
        """
        entry = MemoryEntry(
            content=content,
            timestamp=time.time(),
            importance=importance,
            memory_type=memory_type,
            metadata=metadata or {}
        )
        
        if memory_type == "short_term":
            self.short_term_memory.append(entry)
            # 自动转移重要记忆到长期记忆
            if importance >= 7.0:
                self._promote_to_long_term(entry)
        else:
            self._add_to_long_term(entry)
        
        return entry
    
    def _promote_to_long_term(self, entry: MemoryEntry) -> None:
        """将短期记忆提升为长期记忆 - Promote short-term memory to long-term"""
        entry.memory_type = "long_term"
        self._add_to_long_term(entry)
    
    def _add_to_long_term(self, entry: MemoryEntry) -> None:
        """添加到长期记忆 - Add to long-term memory"""
        if len(self.long_term_memory) >= self.long_term_capacity:
            # 移除最不重要的记忆
            self.long_term_memory.sort(key=lambda x: x.importance)
            self.long_term_memory.pop(0)
        self.long_term_memory.append(entry)
    
    def add_conversation(self, role: str, content: str) -> None:
        """
        添加对话记录
        Add conversation record
        
        Args:
            role: 角色 ("user" or "bot")
            content: 对话内容
        """
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        # 同时添加到短期记忆
        self.add_memory(
            content=f"[{role}]: {content}",
            memory_type="short_term",
            importance=5.0,
            metadata={"type": "conversation", "role": role}
        )
        
        # 更新用户画像
        self._update_user_profile(role)
    
    def _update_user_profile(self, role: str) -> None:
        """更新用户画像 - Update user profile"""
        current_time = time.time()
        self.user_profile["interaction_count"] += 1
        self.user_profile["last_interaction"] = current_time
        
        if self.user_profile["first_interaction"] is None:
            self.user_profile["first_interaction"] = current_time
    
    def get_recent_context(self, n: int = 10) -> List[Dict[str, str]]:
        """
        获取最近的上下文
        Get recent context
        
        Args:
            n: 返回的对话数量
            
        Returns:
            最近n条对话
        """
        return self.conversation_history[-n:] if self.conversation_history else []
    
    def search_memory(
        self,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 5
    ) -> List[MemoryEntry]:
        """
        搜索记忆
        Search memory
        
        Args:
            query: 搜索查询
            memory_type: 记忆类型筛选
            limit: 返回结果数量限制
            
        Returns:
            匹配的记忆条目列表
        """
        results = []
        query_lower = query.lower()
        
        # 搜索短期记忆
        if memory_type is None or memory_type == "short_term":
            for entry in self.short_term_memory:
                if query_lower in entry.content.lower():
                    results.append(entry)
        
        # 搜索长期记忆
        if memory_type is None or memory_type == "long_term":
            for entry in self.long_term_memory:
                if query_lower in entry.content.lower():
                    results.append(entry)
        
        # 按重要性排序
        results.sort(key=lambda x: x.importance, reverse=True)
        return results[:limit]
    
    def add_knowledge(self, key: str, value: Any) -> None:
        """
        添加知识到知识库
        Add knowledge to knowledge base
        
        Args:
            key: 知识键
            value: 知识值
        """
        self.knowledge_base[key] = {
            "value": value,
            "timestamp": time.time()
        }
    
    def get_knowledge(self, key: str) -> Optional[Any]:
        """
        获取知识
        Get knowledge
        
        Args:
            key: 知识键
            
        Returns:
            知识值或None
        """
        if key in self.knowledge_base:
            return self.knowledge_base[key]["value"]
        return None
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        获取记忆摘要
        Get memory summary
        
        Returns:
            记忆统计信息
        """
        return {
            "short_term_count": len(self.short_term_memory),
            "long_term_count": len(self.long_term_memory),
            "knowledge_count": len(self.knowledge_base),
            "conversation_count": len(self.conversation_history),
            "user_profile": self.user_profile
        }
    
    def clear_short_term(self) -> None:
        """清除短期记忆 - Clear short-term memory"""
        self.short_term_memory.clear()
    
    def clear_conversation_history(self) -> None:
        """清除对话历史 - Clear conversation history"""
        self.conversation_history.clear()
    
    def save_to_file(self, filepath: Optional[str] = None) -> None:
        """
        保存记忆到文件
        Save memory to file
        
        Args:
            filepath: 文件路径
        """
        path = filepath or self.persistence_path
        if not path:
            return
        
        data = {
            "short_term_memory": [e.to_dict() for e in self.short_term_memory],
            "long_term_memory": [e.to_dict() for e in self.long_term_memory],
            "knowledge_base": self.knowledge_base,
            "conversation_history": self.conversation_history,
            "user_profile": self.user_profile,
            "saved_at": time.time()
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_from_file(self, filepath: str) -> bool:
        """
        从文件加载记忆
        Load memory from file
        
        Args:
            filepath: 文件路径
            
        Returns:
            是否加载成功
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.short_term_memory = deque(
                [MemoryEntry.from_dict(e) for e in data.get("short_term_memory", [])],
                maxlen=self.short_term_capacity
            )
            self.long_term_memory = [
                MemoryEntry.from_dict(e) for e in data.get("long_term_memory", [])
            ]
            self.knowledge_base = data.get("knowledge_base", {})
            self.conversation_history = data.get("conversation_history", [])
            self.user_profile = data.get("user_profile", self.user_profile)
            
            return True
        except (json.JSONDecodeError, IOError):
            return False
