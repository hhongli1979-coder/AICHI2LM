"""
记忆系统 (Memory System)
========================

实现大模型的记忆能力:
- 短期记忆: 当前对话上下文
- 长期记忆: 持久化的知识和经验
- 工作记忆: 任务执行中的中间状态
- 情节记忆: 交互历史和学习经验
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import OrderedDict
import json
import hashlib

logger = logging.getLogger(__name__)

# Time constants
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400
DAYS_FOR_RELEVANCE = 7


@dataclass
class MemoryItem:
    """记忆项数据类"""
    memory_id: str
    content: Any
    memory_type: str  # 'short_term', 'long_term', 'working', 'episodic'
    importance: float  # 0.0 - 1.0
    access_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'memory_id': self.memory_id,
            'content': self.content,
            'memory_type': self.memory_type,
            'importance': self.importance,
            'access_count': self.access_count,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class ConversationContext:
    """对话上下文数据类"""
    conversation_id: str
    messages: List[Dict[str, str]]
    topic: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: str) -> None:
        """添加消息"""
        self.messages.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        self.last_updated = datetime.now()


@dataclass
class EpisodicMemory:
    """情节记忆数据类"""
    episode_id: str
    event_type: str
    description: str
    outcome: str  # 'success', 'failure', 'neutral'
    lessons_learned: List[str]
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class MemorySystem:
    """
    记忆系统
    
    实现多层次记忆管理:
    1. 短期记忆 - 当前对话上下文，容量有限
    2. 长期记忆 - 持久化的知识和经验
    3. 工作记忆 - 任务执行中的中间状态
    4. 情节记忆 - 交互历史和学习经验
    """
    
    def __init__(
        self,
        short_term_capacity: int = 100,
        working_memory_capacity: int = 50,
        long_term_threshold: float = 0.7
    ):
        """
        初始化记忆系统
        
        Args:
            short_term_capacity: 短期记忆容量
            working_memory_capacity: 工作记忆容量
            long_term_threshold: 转化为长期记忆的重要性阈值
        """
        self.short_term_capacity = short_term_capacity
        self.working_memory_capacity = working_memory_capacity
        self.long_term_threshold = long_term_threshold
        
        # 记忆存储
        self.short_term_memory: OrderedDict[str, MemoryItem] = OrderedDict()
        self.long_term_memory: Dict[str, MemoryItem] = {}
        self.working_memory: OrderedDict[str, MemoryItem] = OrderedDict()
        self.episodic_memory: List[EpisodicMemory] = []
        
        # 对话上下文
        self.current_context: Optional[ConversationContext] = None
        self.context_history: List[ConversationContext] = []
        
        # 统计信息
        self.memory_statistics: Dict[str, int] = {
            'total_stored': 0,
            'total_retrieved': 0,
            'consolidations': 0,
            'forgettings': 0
        }
        
    def store(
        self,
        content: Any,
        memory_type: str = 'short_term',
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        存储记忆
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
            importance: 重要性
            metadata: 元数据
            
        Returns:
            str: 记忆ID
        """
        # 生成记忆ID
        content_hash = hashlib.md5(str(content).encode()).hexdigest()[:8]
        memory_id = f"mem_{memory_type}_{content_hash}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        memory_item = MemoryItem(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata or {}
        )
        
        # 根据类型存储
        if memory_type == 'short_term':
            self._store_short_term(memory_item)
        elif memory_type == 'long_term':
            self._store_long_term(memory_item)
        elif memory_type == 'working':
            self._store_working(memory_item)
            
        self.memory_statistics['total_stored'] += 1
        logger.info(f"Stored memory: {memory_id} (type={memory_type}, importance={importance:.2f})")
        
        return memory_id
        
    def retrieve(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[MemoryItem]:
        """
        检索记忆
        
        Args:
            query: 查询内容
            memory_types: 要搜索的记忆类型
            top_k: 返回的最大数量
            
        Returns:
            List[MemoryItem]: 相关记忆列表
        """
        if memory_types is None:
            memory_types = ['short_term', 'long_term', 'working']
            
        candidates = []
        
        # 从各类记忆中搜索
        if 'short_term' in memory_types:
            candidates.extend(self.short_term_memory.values())
        if 'long_term' in memory_types:
            candidates.extend(self.long_term_memory.values())
        if 'working' in memory_types:
            candidates.extend(self.working_memory.values())
            
        # 计算相关性分数并排序
        scored_memories = []
        for memory in candidates:
            score = self._calculate_relevance(memory, query)
            scored_memories.append((memory, score))
            
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        # 更新访问信息
        results = []
        for memory, _ in scored_memories[:top_k]:
            memory.access_count += 1
            memory.last_accessed = datetime.now()
            results.append(memory)
            
        self.memory_statistics['total_retrieved'] += len(results)
        logger.info(f"Retrieved {len(results)} memories for query: {query[:50]}...")
        
        return results
        
    def consolidate(self) -> int:
        """
        记忆巩固 - 将重要的短期记忆转化为长期记忆
        
        Returns:
            int: 巩固的记忆数量
        """
        consolidated_count = 0
        
        # 找出重要的短期记忆
        important_memories = [
            m for m in self.short_term_memory.values()
            if m.importance >= self.long_term_threshold
        ]
        
        # Collect IDs to remove after iteration
        ids_to_remove = []
        
        for memory in important_memories:
            # Store original ID before modification
            original_id = memory.memory_id
            
            # 转移到长期记忆
            new_id = original_id.replace('short_term', 'long_term')
            memory.memory_id = new_id
            memory.memory_type = 'long_term'
            self.long_term_memory[new_id] = memory
            
            # Mark for removal from short-term memory
            ids_to_remove.append(original_id)
            consolidated_count += 1
        
        # Remove from short-term memory
        for old_id in ids_to_remove:
            if old_id in self.short_term_memory:
                del self.short_term_memory[old_id]
            
        self.memory_statistics['consolidations'] += consolidated_count
        logger.info(f"Consolidated {consolidated_count} memories to long-term storage")
        
        return consolidated_count
        
    def forget(self, decay_rate: float = 0.1) -> int:
        """
        遗忘机制 - 清除不重要或长时间未访问的记忆
        
        Args:
            decay_rate: 衰减率
            
        Returns:
            int: 遗忘的记忆数量
        """
        forgotten_count = 0
        current_time = datetime.now()
        
        # 处理短期记忆
        to_forget = []
        for memory_id, memory in self.short_term_memory.items():
            # 计算时间衰减
            time_delta = (current_time - memory.last_accessed).total_seconds()
            decay = decay_rate * (time_delta / SECONDS_PER_HOUR)  # 每小时衰减
            
            # 根据重要性和访问频率决定是否遗忘
            forget_threshold = memory.importance - decay + (memory.access_count * 0.01)
            
            if forget_threshold < 0.2:
                to_forget.append(memory_id)
                
        for memory_id in to_forget:
            del self.short_term_memory[memory_id]
            forgotten_count += 1
            
        self.memory_statistics['forgettings'] += forgotten_count
        logger.info(f"Forgot {forgotten_count} memories")
        
        return forgotten_count
        
    def start_conversation(self, topic: str = "") -> str:
        """
        开始新对话
        
        Args:
            topic: 对话主题
            
        Returns:
            str: 对话ID
        """
        conversation_id = f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # 保存当前上下文（如果存在）
        if self.current_context:
            self.context_history.append(self.current_context)
            
        # 创建新上下文
        self.current_context = ConversationContext(
            conversation_id=conversation_id,
            messages=[],
            topic=topic
        )
        
        logger.info(f"Started new conversation: {conversation_id}")
        return conversation_id
        
    def add_to_conversation(self, role: str, content: str) -> None:
        """
        添加对话消息
        
        Args:
            role: 角色 ('user', 'assistant', 'system')
            content: 消息内容
        """
        if self.current_context is None:
            self.start_conversation()
            
        self.current_context.add_message(role, content)
        
        # 同时存储到短期记忆
        self.store(
            content={'role': role, 'content': content},
            memory_type='short_term',
            importance=0.6,
            metadata={'conversation_id': self.current_context.conversation_id}
        )
        
    def get_conversation_context(
        self,
        max_messages: int = 10
    ) -> List[Dict[str, str]]:
        """
        获取对话上下文
        
        Args:
            max_messages: 最大消息数
            
        Returns:
            List[Dict[str, str]]: 对话消息列表
        """
        if self.current_context is None:
            return []
            
        return self.current_context.messages[-max_messages:]
        
    def record_episode(
        self,
        event_type: str,
        description: str,
        outcome: str,
        lessons_learned: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        记录情节记忆
        
        Args:
            event_type: 事件类型
            description: 描述
            outcome: 结果
            lessons_learned: 学到的经验
            context: 上下文
            
        Returns:
            str: 情节ID
        """
        episode_id = f"episode_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        episode = EpisodicMemory(
            episode_id=episode_id,
            event_type=event_type,
            description=description,
            outcome=outcome,
            lessons_learned=lessons_learned,
            context=context or {}
        )
        
        self.episodic_memory.append(episode)
        
        # 重要的情节也存储到长期记忆
        if outcome == 'success' or len(lessons_learned) > 0:
            self.store(
                content={
                    'event': event_type,
                    'lessons': lessons_learned
                },
                memory_type='long_term',
                importance=0.8,
                metadata={'episode_id': episode_id}
            )
            
        logger.info(f"Recorded episode: {episode_id} ({outcome})")
        return episode_id
        
    def recall_similar_episodes(
        self,
        event_type: str,
        top_k: int = 3
    ) -> List[EpisodicMemory]:
        """
        回忆相似情节
        
        Args:
            event_type: 事件类型
            top_k: 返回数量
            
        Returns:
            List[EpisodicMemory]: 相似情节列表
        """
        similar_episodes = [
            ep for ep in self.episodic_memory
            if ep.event_type == event_type
        ]
        
        # 按时间倒序排列，返回最近的
        similar_episodes.sort(key=lambda x: x.timestamp, reverse=True)
        
        return similar_episodes[:top_k]
        
    def get_learned_lessons(
        self,
        event_type: Optional[str] = None
    ) -> List[str]:
        """
        获取学习到的经验
        
        Args:
            event_type: 事件类型（可选）
            
        Returns:
            List[str]: 经验列表
        """
        lessons = []
        
        for episode in self.episodic_memory:
            if event_type is None or episode.event_type == event_type:
                lessons.extend(episode.lessons_learned)
                
        # 去重
        return list(set(lessons))
        
    def set_working_memory(
        self,
        key: str,
        value: Any,
        importance: float = 0.7
    ) -> None:
        """
        设置工作记忆
        
        Args:
            key: 键
            value: 值
            importance: 重要性
        """
        memory_id = f"working_{key}"
        
        memory_item = MemoryItem(
            memory_id=memory_id,
            content=value,
            memory_type='working',
            importance=importance,
            metadata={'key': key}
        )
        
        self._store_working(memory_item)
        
    def get_working_memory(self, key: str) -> Optional[Any]:
        """
        获取工作记忆
        
        Args:
            key: 键
            
        Returns:
            Optional[Any]: 值
        """
        memory_id = f"working_{key}"
        
        if memory_id in self.working_memory:
            memory = self.working_memory[memory_id]
            memory.access_count += 1
            memory.last_accessed = datetime.now()
            return memory.content
            
        return None
        
    def clear_working_memory(self) -> None:
        """清除工作记忆"""
        self.working_memory.clear()
        logger.info("Cleared working memory")
        
    def _store_short_term(self, memory_item: MemoryItem) -> None:
        """存储到短期记忆"""
        # 检查容量
        while len(self.short_term_memory) >= self.short_term_capacity:
            # 移除最旧的记忆
            self.short_term_memory.popitem(last=False)
            
        self.short_term_memory[memory_item.memory_id] = memory_item
        
    def _store_long_term(self, memory_item: MemoryItem) -> None:
        """存储到长期记忆"""
        self.long_term_memory[memory_item.memory_id] = memory_item
        
    def _store_working(self, memory_item: MemoryItem) -> None:
        """存储到工作记忆"""
        # 检查容量
        while len(self.working_memory) >= self.working_memory_capacity:
            # 移除最旧的记忆
            self.working_memory.popitem(last=False)
            
        self.working_memory[memory_item.memory_id] = memory_item
        
    def _calculate_relevance(self, memory: MemoryItem, query: str) -> float:
        """
        计算记忆与查询的相关性
        
        Args:
            memory: 记忆项
            query: 查询
            
        Returns:
            float: 相关性分数
        """
        # 简单的关键词匹配实现
        content_str = str(memory.content).lower()
        query_lower = query.lower()
        
        # 基础分数 - 基于重要性
        score = memory.importance * 0.3
        
        # 关键词匹配
        query_words = set(query_lower.split())
        content_words = set(content_str.split())
        overlap = len(query_words & content_words)
        
        if query_words:
            score += 0.4 * (overlap / len(query_words))
            
        # 访问频率加成
        score += min(0.2, memory.access_count * 0.02)
        
        # 时间衰减
        time_delta = (datetime.now() - memory.last_accessed).total_seconds()
        time_factor = max(0, 1 - (time_delta / (SECONDS_PER_DAY * DAYS_FOR_RELEVANCE)))
        score += 0.1 * time_factor
        
        return min(1.0, score)
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取记忆系统统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            'short_term_count': len(self.short_term_memory),
            'long_term_count': len(self.long_term_memory),
            'working_memory_count': len(self.working_memory),
            'episodic_count': len(self.episodic_memory),
            'current_conversation': self.current_context.conversation_id if self.current_context else None,
            'conversation_history_count': len(self.context_history),
            **self.memory_statistics
        }
        
    def export_memory(self) -> Dict[str, Any]:
        """
        导出记忆系统状态
        
        Returns:
            Dict[str, Any]: 记忆系统状态
        """
        return {
            'short_term': [m.to_dict() for m in self.short_term_memory.values()],
            'long_term': [m.to_dict() for m in self.long_term_memory.values()],
            'working': [m.to_dict() for m in self.working_memory.values()],
            'episodic': [
                {
                    'episode_id': ep.episode_id,
                    'event_type': ep.event_type,
                    'description': ep.description,
                    'outcome': ep.outcome,
                    'lessons_learned': ep.lessons_learned,
                    'timestamp': ep.timestamp.isoformat()
                }
                for ep in self.episodic_memory
            ],
            'statistics': self.get_statistics()
        }
        
    def import_memory(self, data: Dict[str, Any]) -> None:
        """
        导入记忆系统状态
        
        Args:
            data: 记忆系统状态数据
        """
        # 导入长期记忆（持久化的重要数据）
        for item in data.get('long_term', []):
            memory_item = MemoryItem(
                memory_id=item['memory_id'],
                content=item['content'],
                memory_type='long_term',
                importance=item['importance'],
                access_count=item['access_count'],
                metadata=item.get('metadata', {})
            )
            self.long_term_memory[memory_item.memory_id] = memory_item
            
        logger.info(f"Imported {len(data.get('long_term', []))} long-term memories")
