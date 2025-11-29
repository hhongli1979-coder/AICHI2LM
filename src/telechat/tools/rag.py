"""
TeleChat RAG 检索增强生成模块
提供知识检索和增强生成功能
"""

import os
import json
import hashlib
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.logging import get_logger
from ..utils.exceptions import TeleChatException, ErrorCodes

logger = get_logger("telechat.rag")


@dataclass
class Document:
    """文档"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata
        }


@dataclass
class RetrievalResult:
    """检索结果"""
    document: Document
    score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "score": self.score
        }


class SimpleVectorStore:
    """简单的向量存储（基于余弦相似度）"""
    
    def __init__(self):
        self.documents: List[Document] = []
        self._embeddings: List[List[float]] = []
    
    def add(self, document: Document, embedding: List[float]):
        """添加文档"""
        document.embedding = embedding
        self.documents.append(document)
        self._embeddings.append(embedding)
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Document, float]]:
        """搜索相似文档"""
        if not self.documents:
            return []
        
        # 计算余弦相似度
        scores = []
        for doc_emb in self._embeddings:
            score = self._cosine_similarity(query_embedding, doc_emb)
            scores.append(score)
        
        # 排序并返回top_k
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in indexed_scores[:top_k]:
            results.append((self.documents[idx], score))
        
        return results
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        if len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def clear(self):
        """清空存储"""
        self.documents.clear()
        self._embeddings.clear()


class TextSplitter:
    """文本分割器"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", ".", " ", ""]
    
    def split(self, text: str) -> List[str]:
        """分割文本"""
        chunks = []
        current_chunk = ""
        
        # 按段落分割
        paragraphs = text.split("\n\n")
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # 如果段落本身就超过chunk_size，需要进一步分割
                if len(para) > self.chunk_size:
                    sub_chunks = self._split_long_text(para)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_long_text(self, text: str) -> List[str]:
        """分割长文本"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # 尝试在分隔符处截断
            best_end = end
            for sep in self.separators:
                if sep:
                    idx = text.rfind(sep, start, end)
                    if idx > start:
                        best_end = idx + len(sep)
                        break
            
            chunks.append(text[start:best_end])
            start = best_end - self.chunk_overlap
        
        return chunks


class SimpleEmbedder:
    """简单的文本嵌入器（基于字符级特征）"""
    
    def __init__(self, dim: int = 128):
        self.dim = dim
    
    def embed(self, text: str) -> List[float]:
        """生成文本嵌入"""
        # 简单的字符级嵌入（实际应用中应使用预训练模型）
        embedding = [0.0] * self.dim
        
        for i, char in enumerate(text):
            idx = ord(char) % self.dim
            embedding[idx] += 1.0
        
        # 归一化
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成嵌入"""
        return [self.embed(text) for text in texts]


class RAGEngine:
    """RAG检索增强生成引擎"""
    
    def __init__(
        self,
        embedder: Optional[SimpleEmbedder] = None,
        vector_store: Optional[SimpleVectorStore] = None
    ):
        self.embedder = embedder or SimpleEmbedder()
        self.vector_store = vector_store or SimpleVectorStore()
        self.splitter = TextSplitter()
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        添加文档到知识库
        
        Args:
            documents: 文档列表
            metadata: 元数据列表
        """
        metadata = metadata or [{}] * len(documents)
        
        for i, (doc_text, meta) in enumerate(zip(documents, metadata)):
            # 分割文档
            chunks = self.splitter.split(doc_text)
            
            for j, chunk in enumerate(chunks):
                # 生成ID
                doc_id = hashlib.md5(f"{i}_{j}_{chunk[:50]}".encode()).hexdigest()[:16]
                
                # 生成嵌入
                embedding = self.embedder.embed(chunk)
                
                # 创建文档
                document = Document(
                    id=doc_id,
                    content=chunk,
                    metadata={**meta, "chunk_index": j}
                )
                
                # 添加到向量存储
                self.vector_store.add(document, embedding)
        
        logger.info(f"添加了 {len(documents)} 个文档到知识库")
    
    def add_text(self, text: str, metadata: Optional[Dict] = None):
        """添加单个文本"""
        self.add_documents([text], [metadata or {}])
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        # 生成查询嵌入
        query_embedding = self.embedder.embed(query)
        
        # 搜索相似文档
        results = self.vector_store.search(query_embedding, top_k)
        
        return [
            RetrievalResult(document=doc, score=score)
            for doc, score in results
        ]
    
    def generate_context(self, query: str, top_k: int = 3) -> str:
        """
        生成增强上下文
        
        Args:
            query: 查询文本
            top_k: 使用的文档数量
            
        Returns:
            增强的上下文文本
        """
        results = self.retrieve(query, top_k)
        
        if not results:
            return ""
        
        context_parts = ["参考信息："]
        for i, result in enumerate(results, 1):
            context_parts.append(f"\n[{i}] {result.document.content}")
        
        return "\n".join(context_parts)
    
    def augmented_prompt(self, query: str, top_k: int = 3) -> str:
        """
        生成增强的提示
        
        Args:
            query: 用户查询
            top_k: 使用的文档数量
            
        Returns:
            增强后的完整提示
        """
        context = self.generate_context(query, top_k)
        
        if context:
            return f"""{context}

根据以上参考信息，回答以下问题：
{query}

回答："""
        else:
            return query
    
    def clear(self):
        """清空知识库"""
        self.vector_store.clear()
        logger.info("知识库已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "document_count": len(self.vector_store.documents),
            "embedding_dim": self.embedder.dim
        }


# 全局RAG引擎
_global_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """获取全局RAG引擎"""
    global _global_rag_engine
    if _global_rag_engine is None:
        _global_rag_engine = RAGEngine()
    return _global_rag_engine
