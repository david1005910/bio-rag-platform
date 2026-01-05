"""PostgreSQL Memory Store Service - Persistent storage for chat memories"""

import logging
import hashlib
from typing import List, Optional
from uuid import UUID
from datetime import datetime, timedelta

from sqlalchemy import select, delete, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.memory import ChatMemory

logger = logging.getLogger(__name__)


class MemoryStoreService:
    """PostgreSQL-based persistent storage for chat memories"""

    def __init__(self, db: AsyncSession):
        self.db = db

    def _hash_query(self, query: str) -> str:
        """Generate a hash for the query for exact match detection"""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    async def save(
        self,
        query: str,
        answer: str,
        sources_used: Optional[List[str]] = None,
        user_id: Optional[UUID] = None,
    ) -> ChatMemory:
        """
        Save a conversation to PostgreSQL

        Args:
            query: User's question
            answer: AI's response
            sources_used: List of PMIDs used in the response
            user_id: Optional user ID

        Returns:
            Created ChatMemory object
        """
        query_hash = self._hash_query(query)
        sources_str = ",".join(sources_used) if sources_used else ""

        memory = ChatMemory(
            query=query,
            answer=answer,
            query_hash=query_hash,
            sources_used=sources_str,
            user_id=user_id,
        )

        self.db.add(memory)
        await self.db.flush()
        await self.db.refresh(memory)

        logger.info(f"Saved memory {memory.id}: '{query[:50]}...'")
        return memory

    async def find_by_hash(self, query_hash: str) -> Optional[ChatMemory]:
        """
        Find a memory by query hash (exact match)

        Args:
            query_hash: MD5 hash of normalized query

        Returns:
            ChatMemory if found, None otherwise
        """
        result = await self.db.execute(
            select(ChatMemory)
            .where(ChatMemory.query_hash == query_hash)
            .order_by(ChatMemory.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def find_exact_match(self, query: str) -> Optional[ChatMemory]:
        """
        Find an exact match for a query

        Args:
            query: User's question

        Returns:
            ChatMemory if found, None otherwise
        """
        query_hash = self._hash_query(query)
        return await self.find_by_hash(query_hash)

    async def search_similar(
        self,
        query: str,
        limit: int = 3,
        min_relevance: float = 0.1
    ) -> List[ChatMemory]:
        """
        Search for similar past conversations using PostgreSQL full-text search

        Args:
            query: User's question
            limit: Maximum number of results
            min_relevance: Minimum relevance score

        Returns:
            List of relevant ChatMemory objects
        """
        # Extract keywords from query (remove common stop words)
        stop_words = {
            '은', '는', '이', '가', '을', '를', '의', '에', '에서', '로', '으로',
            '와', '과', '도', '만', '에게', '한테', '께', '보다', '처럼', '같이',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'what', 'how', 'why', 'when', 'where', 'which', 'who',
            '무엇', '어떻게', '왜', '언제', '어디', '누가', '뭐', '뭘'
        }

        words = query.replace('?', '').replace('!', '').replace('.', '').split()
        keywords = [w for w in words if w.lower() not in stop_words and len(w) > 1]

        if not keywords:
            return []

        # Build ILIKE conditions for keyword matching
        conditions = []
        for keyword in keywords:
            conditions.append(ChatMemory.query.ilike(f"%{keyword}%"))
            conditions.append(ChatMemory.answer.ilike(f"%{keyword}%"))

        # Query with OR conditions
        result = await self.db.execute(
            select(ChatMemory)
            .where(or_(*conditions))
            .order_by(ChatMemory.created_at.desc())
            .limit(limit * 2)  # Get more to filter
        )

        memories = result.scalars().all()

        # Calculate relevance scores
        scored_results = []
        for memory in memories:
            score = 0.0
            query_lower = memory.query.lower()
            answer_lower = memory.answer.lower()

            for keyword in keywords:
                kw_lower = keyword.lower()
                if kw_lower in query_lower:
                    score += 0.3
                if kw_lower in answer_lower:
                    score += 0.2

            if score >= min_relevance:
                memory.relevance_score = min(score, 1.0)
                scored_results.append(memory)

        # Sort by relevance and return top results
        scored_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return scored_results[:limit]

    async def get_recent(
        self,
        limit: int = 10,
        user_id: Optional[UUID] = None
    ) -> List[ChatMemory]:
        """
        Get the most recent conversations

        Args:
            limit: Maximum number of results
            user_id: Optional user ID to filter

        Returns:
            List of recent ChatMemory objects
        """
        query = select(ChatMemory).order_by(ChatMemory.created_at.desc()).limit(limit)

        if user_id:
            query = query.where(ChatMemory.user_id == user_id)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_count(self, user_id: Optional[UUID] = None) -> int:
        """Get the total number of stored memories"""
        query = select(func.count(ChatMemory.id))

        if user_id:
            query = query.where(ChatMemory.user_id == user_id)

        result = await self.db.execute(query)
        return result.scalar() or 0

    async def delete_old(self, days: int = 30) -> int:
        """
        Delete memories older than specified days

        Args:
            days: Number of days to keep

        Returns:
            Number of deleted memories
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        result = await self.db.execute(
            delete(ChatMemory).where(ChatMemory.created_at < cutoff_date)
        )

        deleted_count = result.rowcount
        logger.info(f"Deleted {deleted_count} old memories (older than {days} days)")
        return deleted_count

    async def delete_by_id(self, memory_id: UUID) -> bool:
        """Delete a specific memory by ID"""
        result = await self.db.execute(
            delete(ChatMemory).where(ChatMemory.id == memory_id)
        )
        return result.rowcount > 0

    async def get_by_id(self, memory_id: UUID) -> Optional[ChatMemory]:
        """Get a memory by ID"""
        result = await self.db.execute(
            select(ChatMemory).where(ChatMemory.id == memory_id)
        )
        return result.scalar_one_or_none()
