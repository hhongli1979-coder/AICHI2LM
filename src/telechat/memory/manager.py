"""
TeleChat 记忆系统模块
提供短期记忆和长期记忆管理
"""

import json
import sqlite3
import time
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import hashlib

from ..utils.config import get_config, MemoryConfig
from ..utils.logging import get_logger
from ..utils.exceptions import MemoryException, ErrorCodes

logger = get_logger("telechat.memory")


@dataclass
class MemoryItem:
    """记忆项"""
    id: str
    content: str
    role: str  # user/bot
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    importance: float = 0.5  # 重要性评分 0-1
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "role": self.role,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "access_count": self.access_count,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        return cls(**data)


class ShortTermMemory:
    """短期记忆（工作记忆）"""
    
    def __init__(self, capacity: int = 100):
        """
        初始化短期记忆
        
        Args:
            capacity: 最大容量
        """
        self.capacity = capacity
        self._memory: Dict[str, List[MemoryItem]] = {}  # session_id -> items
        
    def add(self, item: MemoryItem):
        """添加记忆项"""
        session_id = item.session_id or "default"
        
        if session_id not in self._memory:
            self._memory[session_id] = []
        
        self._memory[session_id].append(item)
        
        # 容量限制
        if len(self._memory[session_id]) > self.capacity:
            # 移除最旧的记忆
            self._memory[session_id] = self._memory[session_id][-self.capacity:]
    
    def get(self, session_id: str, limit: int = 10) -> List[MemoryItem]:
        """获取指定会话的记忆"""
        if session_id not in self._memory:
            return []
        return self._memory[session_id][-limit:]
    
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """获取对话历史（格式化为模型输入）"""
        items = self.get(session_id)
        return [
            {"role": item.role, "content": item.content}
            for item in items
        ]
    
    def clear(self, session_id: Optional[str] = None):
        """清除记忆"""
        if session_id:
            self._memory.pop(session_id, None)
        else:
            self._memory.clear()
    
    def get_all_sessions(self) -> List[str]:
        """获取所有会话ID"""
        return list(self._memory.keys())


class LongTermMemory:
    """长期记忆（持久化存储）"""
    
    def __init__(self, db_path: str = "telechat_memory.db"):
        """
        初始化长期记忆
        
        Args:
            db_path: SQLite数据库路径
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建记忆表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                role TEXT NOT NULL,
                user_id TEXT,
                session_id TEXT,
                timestamp TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_id ON memories(user_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_id ON memories(session_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)
        """)
        
        # 创建用户画像表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                preferences TEXT,
                entities TEXT,
                summary TEXT,
                last_interaction TEXT,
                interaction_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"长期记忆数据库初始化完成: {self.db_path}")
    
    def store(self, item: MemoryItem):
        """存储记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO memories 
                (id, content, role, user_id, session_id, timestamp, importance, access_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.id,
                item.content,
                item.role,
                item.user_id,
                item.session_id,
                item.timestamp,
                item.importance,
                item.access_count,
                json.dumps(item.metadata)
            ))
            conn.commit()
            
        except Exception as e:
            logger.error(f"存储记忆失败: {str(e)}")
            raise MemoryException(
                message=f"存储记忆失败: {str(e)}",
                code=ErrorCodes.MEMORY_ERROR
            )
        finally:
            conn.close()
    
    def retrieve(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 50,
        since: Optional[str] = None
    ) -> List[MemoryItem]:
        """检索记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            query = "SELECT * FROM memories WHERE 1=1"
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            
            if since:
                query += " AND timestamp >= ?"
                params.append(since)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            items = []
            for row in rows:
                items.append(MemoryItem(
                    id=row[0],
                    content=row[1],
                    role=row[2],
                    user_id=row[3],
                    session_id=row[4],
                    timestamp=row[5],
                    importance=row[6],
                    access_count=row[7],
                    metadata=json.loads(row[8]) if row[8] else {}
                ))
            
            return items
            
        except Exception as e:
            logger.error(f"检索记忆失败: {str(e)}")
            raise MemoryException(
                message=f"检索记忆失败: {str(e)}",
                code=ErrorCodes.MEMORY_RETRIEVE_ERROR
            )
        finally:
            conn.close()
    
    def search(self, query: str, user_id: Optional[str] = None, limit: int = 10) -> List[MemoryItem]:
        """搜索记忆（简单文本匹配）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            sql = "SELECT * FROM memories WHERE content LIKE ?"
            params = [f"%{query}%"]
            
            if user_id:
                sql += " AND user_id = ?"
                params.append(user_id)
            
            sql += " ORDER BY importance DESC, timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            items = []
            for row in rows:
                items.append(MemoryItem(
                    id=row[0],
                    content=row[1],
                    role=row[2],
                    user_id=row[3],
                    session_id=row[4],
                    timestamp=row[5],
                    importance=row[6],
                    access_count=row[7],
                    metadata=json.loads(row[8]) if row[8] else {}
                ))
            
            return items
            
        finally:
            conn.close()
    
    def update_importance(self, memory_id: str, importance: float):
        """更新记忆重要性"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE memories SET importance = ? WHERE id = ?
            """, (importance, memory_id))
            conn.commit()
        finally:
            conn.close()
    
    def increment_access(self, memory_id: str):
        """增加访问计数"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE memories SET access_count = access_count + 1 WHERE id = ?
            """, (memory_id,))
            conn.commit()
        finally:
            conn.close()
    
    def delete(self, memory_id: str):
        """删除记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
        finally:
            conn.close()
    
    def clear_user_memories(self, user_id: str):
        """清除用户所有记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
            conn.commit()
        finally:
            conn.close()
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户画像"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return {
                    "user_id": row[0],
                    "preferences": json.loads(row[1]) if row[1] else {},
                    "entities": json.loads(row[2]) if row[2] else [],
                    "summary": row[3],
                    "last_interaction": row[4],
                    "interaction_count": row[5]
                }
            return None
            
        finally:
            conn.close()
    
    def update_user_profile(self, user_id: str, profile: Dict[str, Any]):
        """更新用户画像"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO user_profiles 
                (user_id, preferences, entities, summary, last_interaction, interaction_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                user_id,
                json.dumps(profile.get("preferences", {})),
                json.dumps(profile.get("entities", [])),
                profile.get("summary", ""),
                profile.get("last_interaction", datetime.now().isoformat()),
                profile.get("interaction_count", 1)
            ))
            conn.commit()
        finally:
            conn.close()


class MemoryManager:
    """记忆管理器"""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        初始化记忆管理器
        
        Args:
            config: 记忆配置
        """
        self.config = config or get_config().memory
        
        # 初始化短期记忆
        self.short_term = ShortTermMemory(
            capacity=self.config.short_term_capacity
        )
        
        # 初始化长期记忆
        if self.config.long_term_enabled:
            db_url = self.config.database_url
            if db_url.startswith("sqlite:///"):
                db_path = db_url[10:]
            else:
                db_path = "telechat_memory.db"
            self.long_term = LongTermMemory(db_path)
        else:
            self.long_term = None
        
        logger.info("记忆管理器初始化完成")
    
    def _generate_id(self, content: str) -> str:
        """生成记忆ID"""
        timestamp = str(time.time())
        return hashlib.md5(f"{content}{timestamp}".encode()).hexdigest()[:16]
    
    def add_message(
        self,
        content: str,
        role: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryItem:
        """
        添加消息到记忆
        
        Args:
            content: 消息内容
            role: 角色 (user/bot)
            user_id: 用户ID
            session_id: 会话ID
            importance: 重要性评分
            metadata: 额外元数据
            
        Returns:
            创建的记忆项
        """
        item = MemoryItem(
            id=self._generate_id(content),
            content=content,
            role=role,
            user_id=user_id,
            session_id=session_id,
            importance=importance,
            metadata=metadata or {}
        )
        
        # 添加到短期记忆
        self.short_term.add(item)
        
        # 如果启用长期记忆且重要性足够高，则持久化
        if self.long_term and importance >= 0.7:
            self.long_term.store(item)
        
        return item
    
    def get_context(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        max_history: int = 10,
        include_long_term: bool = True
    ) -> List[Dict[str, str]]:
        """
        获取对话上下文
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            max_history: 最大历史条数
            include_long_term: 是否包含长期记忆
            
        Returns:
            对话历史列表
        """
        # 从短期记忆获取
        history = self.short_term.get_history(session_id)
        
        # 如果短期记忆不足，从长期记忆补充
        if include_long_term and self.long_term and len(history) < max_history:
            long_term_items = self.long_term.retrieve(
                user_id=user_id,
                session_id=session_id,
                limit=max_history - len(history)
            )
            
            for item in reversed(long_term_items):
                history.insert(0, {"role": item.role, "content": item.content})
        
        return history[-max_history:]
    
    def search_memories(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 5
    ) -> List[MemoryItem]:
        """搜索相关记忆"""
        if self.long_term:
            return self.long_term.search(query, user_id, limit)
        return []
    
    def clear_session(self, session_id: str):
        """清除会话记忆"""
        self.short_term.clear(session_id)
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户画像"""
        if self.long_term:
            return self.long_term.get_user_profile(user_id)
        return None
    
    def update_user_profile(self, user_id: str, profile: Dict[str, Any]):
        """更新用户画像"""
        if self.long_term:
            self.long_term.update_user_profile(user_id, profile)


# 全局记忆管理器
_global_memory: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """获取全局记忆管理器"""
    global _global_memory
    if _global_memory is None:
        _global_memory = MemoryManager()
    return _global_memory
