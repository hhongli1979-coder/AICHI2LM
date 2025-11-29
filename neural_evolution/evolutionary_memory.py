# -*- coding: utf-8 -*-
"""
进化记忆系统 - Evolutionary Memory System

实现上下文进化和记忆优化机制。
This implements context evolution and memory optimization mechanisms.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Insight:
    """洞察/知识单元"""
    topic: str
    content: str
    confidence: float
    source: str
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def access(self) -> None:
        """记录访问"""
        self.access_count += 1
        self.last_accessed = time.time()


@dataclass
class Experience:
    """经验记录"""
    task_type: str
    input_data: Any
    output_data: Any
    success: bool
    score: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvolutionaryMemory:
    """
    进化记忆系统 - Evolutionary Memory System

    实现短期记忆和长期记忆的管理，以及从经验中提炼知识的能力。
    Manages short-term and long-term memory, with ability to
    extract insights from experiences.
    """

    def __init__(
        self,
        short_term_capacity: int = 100,
        long_term_capacity: int = 10000,
        consolidation_threshold: int = 5
    ):
        """
        初始化进化记忆系统

        Args:
            short_term_capacity: 短期记忆容量
            long_term_capacity: 长期记忆容量
            consolidation_threshold: 记忆固化阈值（访问次数）
        """
        self.short_term: List[Experience] = []
        self.long_term: Dict[str, Insight] = {}
        self.topic_index: Dict[str, List[str]] = defaultdict(list)

        self.short_term_capacity = short_term_capacity
        self.long_term_capacity = long_term_capacity
        self.consolidation_threshold = consolidation_threshold

        self._statistics = {
            "total_experiences": 0,
            "total_insights": 0,
            "consolidations": 0,
            "evictions": 0
        }

    def add_experience(self, experience: Experience) -> None:
        """
        添加新经验到短期记忆

        Args:
            experience: 经验记录
        """
        self.short_term.append(experience)
        self._statistics["total_experiences"] += 1

        # 如果超出容量，移除最旧的经验
        if len(self.short_term) > self.short_term_capacity:
            self._evict_short_term()

    def _evict_short_term(self) -> None:
        """从短期记忆中驱逐最旧的经验"""
        if self.short_term:
            self.short_term.pop(0)
            self._statistics["evictions"] += 1

    def extract_insight(self, experience: Experience) -> Optional[Insight]:
        """
        从经验中提炼洞察/知识

        Args:
            experience: 经验记录

        Returns:
            提炼出的洞察，如果无法提炼则返回None
        """
        if not experience.success or experience.score < 0.7:
            return None

        # 生成洞察内容
        insight_content = self._generate_insight_content(experience)
        if not insight_content:
            return None

        insight = Insight(
            topic=experience.task_type,
            content=insight_content,
            confidence=experience.score,
            source=f"experience_{experience.timestamp}"
        )

        return insight

    def _generate_insight_content(self, experience: Experience) -> str:
        """
        从经验生成洞察内容

        Args:
            experience: 经验记录

        Returns:
            洞察内容字符串
        """
        # 基于经验元数据生成洞察
        metadata = experience.metadata
        task_type = experience.task_type

        if "strategy" in metadata:
            return f"Strategy '{metadata['strategy']}' is effective for {task_type} tasks"
        elif "approach" in metadata:
            return f"Approach '{metadata['approach']}' works well for {task_type}"
        else:
            return f"Successfully completed {task_type} task with score {experience.score}"

    def evolve_memory(self, experience: Experience) -> Optional[Insight]:
        """
        进化记忆 - 从经验中学习并更新长期记忆

        Args:
            experience: 经验记录

        Returns:
            新创建或更新的洞察
        """
        # 添加到短期记忆
        self.add_experience(experience)

        # 尝试提炼洞察
        insight = self.extract_insight(experience)
        if insight is None:
            return None

        # 更新长期记忆
        return self._update_long_term(insight)

    def _update_long_term(self, insight: Insight) -> Insight:
        """
        更新长期记忆

        Args:
            insight: 新洞察

        Returns:
            更新后的洞察
        """
        key = f"{insight.topic}_{hash(insight.content) % 10000}"

        if key in self.long_term:
            # 更新现有洞察的置信度
            existing = self.long_term[key]
            existing.confidence = (existing.confidence + insight.confidence) / 2
            existing.access()
            return existing
        else:
            # 添加新洞察
            if len(self.long_term) >= self.long_term_capacity:
                self._evict_long_term()

            self.long_term[key] = insight
            self.topic_index[insight.topic].append(key)
            self._statistics["total_insights"] += 1
            self._statistics["consolidations"] += 1
            return insight

    def _evict_long_term(self) -> None:
        """从长期记忆中驱逐最少使用的洞察"""
        if not self.long_term:
            return

        # 找到访问次数最少且置信度最低的洞察
        least_used_key = min(
            self.long_term.keys(),
            key=lambda k: (
                self.long_term[k].access_count,
                self.long_term[k].confidence
            )
        )

        insight = self.long_term.pop(least_used_key)
        if insight.topic in self.topic_index:
            if least_used_key in self.topic_index[insight.topic]:
                self.topic_index[insight.topic].remove(least_used_key)
        self._statistics["evictions"] += 1

    def query_insights(
        self,
        topic: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 10
    ) -> List[Insight]:
        """
        查询长期记忆中的洞察

        Args:
            topic: 主题过滤
            min_confidence: 最低置信度
            limit: 返回数量限制

        Returns:
            匹配的洞察列表
        """
        if topic:
            keys = self.topic_index.get(topic, [])
            insights = [self.long_term[k] for k in keys if k in self.long_term]
        else:
            insights = list(self.long_term.values())

        # 过滤并排序
        filtered = [i for i in insights if i.confidence >= min_confidence]
        sorted_insights = sorted(
            filtered,
            key=lambda x: (x.confidence, x.access_count),
            reverse=True
        )

        # 更新访问记录
        result = sorted_insights[:limit]
        for insight in result:
            insight.access()

        return result

    def get_relevant_context(
        self,
        task_type: str,
        max_insights: int = 5,
        max_experiences: int = 3
    ) -> Dict[str, Any]:
        """
        获取与任务相关的上下文

        Args:
            task_type: 任务类型
            max_insights: 最大洞察数
            max_experiences: 最大经验数

        Returns:
            相关上下文字典
        """
        # 获取相关洞察
        relevant_insights = self.query_insights(
            topic=task_type,
            min_confidence=0.5,
            limit=max_insights
        )

        # 获取相关经验
        relevant_experiences = [
            exp for exp in reversed(self.short_term)
            if exp.task_type == task_type and exp.success
        ][:max_experiences]

        return {
            "insights": [
                {"topic": i.topic, "content": i.content, "confidence": i.confidence}
                for i in relevant_insights
            ],
            "recent_experiences": [
                {"score": e.score, "metadata": e.metadata}
                for e in relevant_experiences
            ]
        }

    def consolidate_memories(self) -> int:
        """
        记忆固化 - 将频繁访问的短期记忆转化为长期知识

        Returns:
            固化的洞察数量
        """
        consolidated = 0

        # 分析短期记忆中的模式
        task_patterns: Dict[str, List[Experience]] = defaultdict(list)
        for exp in self.short_term:
            if exp.success:
                task_patterns[exp.task_type].append(exp)

        # 为每种任务类型创建洞察
        for task_type, experiences in task_patterns.items():
            if len(experiences) >= self.consolidation_threshold:
                avg_score = sum(e.score for e in experiences) / len(experiences)
                insight = Insight(
                    topic=task_type,
                    content=f"Pattern identified: {len(experiences)} successful {task_type} tasks with avg score {avg_score:.2f}",
                    confidence=avg_score,
                    source="consolidation"
                )
                self._update_long_term(insight)
                consolidated += 1

        return consolidated

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取记忆系统统计信息

        Returns:
            统计信息字典
        """
        return {
            **self._statistics,
            "short_term_count": len(self.short_term),
            "long_term_count": len(self.long_term),
            "topics": list(self.topic_index.keys()),
            "short_term_capacity": self.short_term_capacity,
            "long_term_capacity": self.long_term_capacity
        }

    def clear_short_term(self) -> None:
        """清空短期记忆"""
        self.short_term.clear()

    def export_insights(self) -> List[Dict[str, Any]]:
        """
        导出所有洞察

        Returns:
            洞察列表
        """
        return [
            {
                "topic": insight.topic,
                "content": insight.content,
                "confidence": insight.confidence,
                "source": insight.source,
                "access_count": insight.access_count
            }
            for insight in self.long_term.values()
        ]

    def import_insights(self, insights_data: List[Dict[str, Any]]) -> int:
        """
        导入洞察

        Args:
            insights_data: 洞察数据列表

        Returns:
            成功导入的数量
        """
        imported = 0
        for data in insights_data:
            insight = Insight(
                topic=data.get("topic", "unknown"),
                content=data.get("content", ""),
                confidence=data.get("confidence", 0.5),
                source=data.get("source", "import")
            )
            self._update_long_term(insight)
            imported += 1

        return imported
