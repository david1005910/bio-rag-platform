"""Chat Memory Service - Redis Cache + PostgreSQL Persistent Storage

Architecture:
- Redis: Fast cache layer for exact match lookups and recent queries
- PostgreSQL: Persistent storage for all conversation history

Flow:
1. Search: Check Redis cache first -> If miss, query PostgreSQL -> Cache result
2. Save: Save to PostgreSQL -> Cache in Redis for quick subsequent lookups
"""

import logging
import hashlib
from typing import List, Optional
from uuid import UUID
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from src.services.redis_cache import RedisCacheService, get_redis_cache
from src.services.memory_store import MemoryStoreService
from src.models.memory import ChatMemory

logger = logging.getLogger(__name__)


@dataclass
class ConversationMemory:
    """Stored conversation memory"""
    id: str
    query: str
    answer: str
    query_hash: str
    created_at: str
    relevance_score: float = 0.0
    sources_used: Optional[str] = None


class ChatMemoryService:
    """
    Hybrid chat memory service using Redis cache + PostgreSQL storage

    Features:
    - Fast exact match lookups via Redis cache
    - Persistent storage in PostgreSQL
    - Automatic cache invalidation on updates
    - Fallback to PostgreSQL if Redis is unavailable
    """

    def __init__(
        self,
        db: AsyncSession,
        redis_cache: Optional[RedisCacheService] = None
    ):
        self.db = db
        self.store = MemoryStoreService(db)
        self._redis = redis_cache
        self._redis_available = True

    async def _get_redis(self) -> Optional[RedisCacheService]:
        """Get Redis cache service, handling connection failures gracefully"""
        if not self._redis_available:
            return None

        try:
            if self._redis is None:
                self._redis = await get_redis_cache()

            # Check connection
            if await self._redis.is_connected():
                return self._redis
            else:
                self._redis_available = False
                logger.warning("Redis not available, falling back to PostgreSQL only")
                return None

        except Exception as e:
            self._redis_available = False
            logger.warning(f"Redis connection failed: {e}, falling back to PostgreSQL only")
            return None

    def _hash_query(self, query: str) -> str:
        """Generate a hash for the query for exact match detection"""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def _memory_to_dataclass(self, memory: ChatMemory) -> ConversationMemory:
        """Convert ChatMemory model to ConversationMemory dataclass"""
        return ConversationMemory(
            id=str(memory.id),
            query=memory.query,
            answer=memory.answer,
            query_hash=memory.query_hash,
            created_at=memory.created_at.isoformat() if memory.created_at else "",
            relevance_score=memory.relevance_score or 0.0,
            sources_used=memory.sources_used,
        )

    def _dict_to_dataclass(self, data: dict) -> ConversationMemory:
        """Convert dict to ConversationMemory dataclass"""
        return ConversationMemory(
            id=data.get("id", ""),
            query=data.get("query", ""),
            answer=data.get("answer", ""),
            query_hash=data.get("query_hash", ""),
            created_at=data.get("created_at", ""),
            relevance_score=data.get("relevance_score", 0.0),
            sources_used=data.get("sources_used"),
        )

    async def save_conversation(
        self,
        query: str,
        answer: str,
        sources_used: Optional[List[str]] = None,
        user_id: Optional[UUID] = None,
    ) -> str:
        """
        Save a conversation to PostgreSQL and cache in Redis

        Args:
            query: User's question
            answer: AI's response
            sources_used: List of PMIDs used in the response
            user_id: Optional user ID

        Returns:
            The ID of the saved conversation
        """
        # Save to PostgreSQL (persistent storage)
        memory = await self.store.save(
            query=query,
            answer=answer,
            sources_used=sources_used,
            user_id=user_id,
        )

        # Cache in Redis for fast subsequent lookups
        redis = await self._get_redis()
        if redis:
            try:
                await redis.set_by_hash(
                    memory.query_hash,
                    memory.to_dict()
                )
                # Invalidate recent cache since we added new data
                await redis.invalidate_recent()
                logger.debug(f"Cached memory in Redis: {memory.query_hash[:8]}...")
            except Exception as e:
                logger.warning(f"Failed to cache in Redis: {e}")

        logger.info(f"Saved conversation {memory.id}: '{query[:50]}...'")
        return str(memory.id)

    async def find_exact_match(self, query: str) -> Optional[ConversationMemory]:
        """
        Find an exact match for a query

        Flow: Redis cache -> PostgreSQL

        Args:
            query: User's question

        Returns:
            ConversationMemory if found, None otherwise
        """
        query_hash = self._hash_query(query)

        # 1. Try Redis cache first
        redis = await self._get_redis()
        if redis:
            try:
                cached = await redis.get_by_hash(query_hash)
                if cached:
                    logger.info(f"Cache HIT for exact match: {query[:30]}...")
                    return self._dict_to_dataclass(cached)
            except Exception as e:
                logger.warning(f"Redis lookup failed: {e}")

        # 2. Fall back to PostgreSQL
        memory = await self.store.find_by_hash(query_hash)

        if memory:
            logger.info(f"PostgreSQL HIT for exact match: {query[:30]}...")

            # Cache result in Redis for next time
            if redis:
                try:
                    await redis.set_by_hash(query_hash, memory.to_dict())
                except Exception as e:
                    logger.warning(f"Failed to cache result: {e}")

            return self._memory_to_dataclass(memory)

        logger.debug(f"No exact match found for: {query[:30]}...")
        return None

    async def search_similar_conversations(
        self,
        query: str,
        limit: int = 3,
        min_relevance: float = 0.1
    ) -> List[ConversationMemory]:
        """
        Search for similar past conversations

        Flow: Check Redis cache -> PostgreSQL search

        Args:
            query: User's question
            limit: Maximum number of results
            min_relevance: Minimum relevance score

        Returns:
            List of relevant ConversationMemory objects
        """
        # Check Redis cache for search results
        redis = await self._get_redis()
        if redis:
            try:
                cached = await redis.get_search_results(query)
                if cached:
                    logger.info(f"Cache HIT for similar search: {query[:30]}...")
                    return [self._dict_to_dataclass(m) for m in cached]
            except Exception as e:
                logger.warning(f"Redis search cache lookup failed: {e}")

        # Search in PostgreSQL
        memories = await self.store.search_similar(
            query=query,
            limit=limit,
            min_relevance=min_relevance
        )

        results = [self._memory_to_dataclass(m) for m in memories]

        # Cache search results in Redis
        if redis and results:
            try:
                await redis.set_search_results(
                    query,
                    [
                        {
                            "id": r.id,
                            "query": r.query,
                            "answer": r.answer,
                            "query_hash": r.query_hash,
                            "created_at": r.created_at,
                            "relevance_score": r.relevance_score,
                            "sources_used": r.sources_used,
                        }
                        for r in results
                    ]
                )
            except Exception as e:
                logger.warning(f"Failed to cache search results: {e}")

        return results

    async def get_recent_conversations(
        self,
        limit: int = 10,
        user_id: Optional[UUID] = None
    ) -> List[ConversationMemory]:
        """
        Get the most recent conversations

        Flow: Redis cache -> PostgreSQL

        Args:
            limit: Maximum number of results
            user_id: Optional user ID to filter

        Returns:
            List of recent ConversationMemory objects
        """
        # Try Redis cache first (only for non-user-specific queries)
        if not user_id:
            redis = await self._get_redis()
            if redis:
                try:
                    cached = await redis.get_recent(limit)
                    if cached:
                        logger.debug("Cache HIT for recent conversations")
                        return [self._dict_to_dataclass(m) for m in cached]
                except Exception as e:
                    logger.warning(f"Redis recent cache lookup failed: {e}")

        # Fetch from PostgreSQL
        memories = await self.store.get_recent(limit=limit, user_id=user_id)
        results = [self._memory_to_dataclass(m) for m in memories]

        # Cache results in Redis (only for non-user-specific queries)
        if not user_id:
            redis = await self._get_redis()
            if redis and results:
                try:
                    await redis.set_recent([
                        {
                            "id": r.id,
                            "query": r.query,
                            "answer": r.answer,
                            "query_hash": r.query_hash,
                            "created_at": r.created_at,
                            "relevance_score": r.relevance_score,
                            "sources_used": r.sources_used,
                        }
                        for r in results
                    ])
                except Exception as e:
                    logger.warning(f"Failed to cache recent: {e}")

        return results

    async def get_conversation_count(self, user_id: Optional[UUID] = None) -> int:
        """Get the total number of stored conversations"""
        return await self.store.get_count(user_id=user_id)

    async def clear_old_conversations(self, days: int = 30) -> int:
        """
        Clear conversations older than specified days

        Args:
            days: Number of days to keep

        Returns:
            Number of deleted conversations
        """
        deleted = await self.store.delete_old(days=days)

        # Clear Redis cache since data has changed
        redis = await self._get_redis()
        if redis:
            try:
                await redis.clear_all_memory_cache()
            except Exception as e:
                logger.warning(f"Failed to clear Redis cache: {e}")

        return deleted

    async def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        stats = {
            "postgresql_count": await self.store.get_count(),
            "redis_available": self._redis_available,
        }

        redis = await self._get_redis()
        if redis:
            try:
                redis_stats = await redis.get_cache_stats()
                stats["redis"] = redis_stats
            except Exception as e:
                stats["redis"] = {"error": str(e)}

        return stats

    @property
    def db_path(self) -> str:
        """Return database type for compatibility"""
        return "PostgreSQL + Redis"


# ============== Helper Functions ==============

def format_memory_context(memories: List[ConversationMemory]) -> str:
    """
    Format past conversations for inclusion in AI prompt

    Args:
        memories: List of relevant past conversations

    Returns:
        Formatted string for prompt injection
    """
    if not memories:
        return ""

    context_parts = ["\n\n📚 **관련 과거 대화 (Related Past Conversations):**\n"]
    context_parts.append("이전에 유사한 질문에 대해 답변한 내용입니다. 참고하여 일관성 있는 답변을 제공해주세요.\n")

    for i, memory in enumerate(memories, 1):
        # Truncate long answers for context
        answer_preview = memory.answer[:500] + "..." if len(memory.answer) > 500 else memory.answer

        context_parts.append(f"""
---
**과거 질문 {i}**: {memory.query}
**과거 답변 요약**: {answer_preview}
**관련도**: {memory.relevance_score:.2f}
""")

    context_parts.append("\n---\n위의 과거 대화를 참고하되, 새로운 정보가 있다면 업데이트된 내용으로 답변해주세요.\n")

    return "".join(context_parts)


# ============== Dependency Injection ==============

async def get_memory_service(db: AsyncSession) -> ChatMemoryService:
    """
    Get chat memory service instance

    Usage:
        @app.get("/chat")
        async def chat(
            memory: ChatMemoryService = Depends(get_memory_service_dep)
        ):
            ...
    """
    redis = await get_redis_cache()
    return ChatMemoryService(db=db, redis_cache=redis)
