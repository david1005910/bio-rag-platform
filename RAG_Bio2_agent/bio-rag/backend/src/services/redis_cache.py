"""Redis Cache Service for chat memory caching"""

import json
import logging
from typing import Optional
import redis.asyncio as redis

from src.core.config import settings

logger = logging.getLogger(__name__)


class RedisCacheService:
    """Redis-based caching service for fast lookups"""

    # Cache key prefixes
    PREFIX_QUERY_HASH = "memory:hash:"  # For exact match lookups
    PREFIX_QUERY_RESULT = "memory:result:"  # For search result caching
    PREFIX_RECENT = "memory:recent"  # For recent conversations

    # Default TTL values (in seconds)
    TTL_QUERY_HASH = 3600 * 24  # 24 hours for exact match cache
    TTL_QUERY_RESULT = 3600  # 1 hour for search results
    TTL_RECENT = 3600 * 6  # 6 hours for recent conversations

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self._client: Optional[redis.Redis] = None

    async def get_client(self) -> redis.Redis:
        """Get or create Redis client"""
        if self._client is None:
            try:
                self._client = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                # Test connection
                await self._client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                raise
        return self._client

    async def close(self):
        """Close Redis connection"""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Redis connection closed")

    # ============== Query Hash Cache (Exact Match) ==============

    async def get_by_hash(self, query_hash: str) -> Optional[dict]:
        """
        Get cached memory by query hash (exact match)

        Args:
            query_hash: MD5 hash of normalized query

        Returns:
            Cached memory dict or None
        """
        try:
            client = await self.get_client()
            key = f"{self.PREFIX_QUERY_HASH}{query_hash}"
            data = await client.get(key)

            if data:
                logger.debug(f"Cache HIT for hash: {query_hash[:8]}...")
                return json.loads(data)

            logger.debug(f"Cache MISS for hash: {query_hash[:8]}...")
            return None

        except Exception as e:
            logger.warning(f"Redis get_by_hash error: {e}")
            return None

    async def set_by_hash(
        self,
        query_hash: str,
        memory_data: dict,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache memory by query hash

        Args:
            query_hash: MD5 hash of normalized query
            memory_data: Memory data to cache
            ttl: Time to live in seconds

        Returns:
            Success status
        """
        try:
            client = await self.get_client()
            key = f"{self.PREFIX_QUERY_HASH}{query_hash}"

            await client.setex(
                key,
                ttl or self.TTL_QUERY_HASH,
                json.dumps(memory_data, ensure_ascii=False)
            )

            logger.debug(f"Cached memory for hash: {query_hash[:8]}...")
            return True

        except Exception as e:
            logger.warning(f"Redis set_by_hash error: {e}")
            return False

    async def invalidate_hash(self, query_hash: str) -> bool:
        """Invalidate cache for a query hash"""
        try:
            client = await self.get_client()
            key = f"{self.PREFIX_QUERY_HASH}{query_hash}"
            await client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis invalidate_hash error: {e}")
            return False

    # ============== Search Result Cache ==============

    async def get_search_results(self, query: str) -> Optional[list]:
        """Get cached search results for a query"""
        try:
            client = await self.get_client()
            key = f"{self.PREFIX_QUERY_RESULT}{hash(query)}"
            data = await client.get(key)

            if data:
                logger.debug(f"Search cache HIT for: {query[:30]}...")
                return json.loads(data)
            return None

        except Exception as e:
            logger.warning(f"Redis get_search_results error: {e}")
            return None

    async def set_search_results(
        self,
        query: str,
        results: list,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache search results for a query"""
        try:
            client = await self.get_client()
            key = f"{self.PREFIX_QUERY_RESULT}{hash(query)}"

            await client.setex(
                key,
                ttl or self.TTL_QUERY_RESULT,
                json.dumps(results, ensure_ascii=False)
            )
            return True

        except Exception as e:
            logger.warning(f"Redis set_search_results error: {e}")
            return False

    # ============== Recent Conversations Cache ==============

    async def get_recent(self, limit: int = 10) -> Optional[list]:
        """Get cached recent conversations"""
        try:
            client = await self.get_client()
            data = await client.get(self.PREFIX_RECENT)

            if data:
                results = json.loads(data)
                return results[:limit]
            return None

        except Exception as e:
            logger.warning(f"Redis get_recent error: {e}")
            return None

    async def set_recent(self, memories: list, ttl: Optional[int] = None) -> bool:
        """Cache recent conversations"""
        try:
            client = await self.get_client()

            await client.setex(
                self.PREFIX_RECENT,
                ttl or self.TTL_RECENT,
                json.dumps(memories, ensure_ascii=False)
            )
            return True

        except Exception as e:
            logger.warning(f"Redis set_recent error: {e}")
            return False

    async def invalidate_recent(self) -> bool:
        """Invalidate recent conversations cache"""
        try:
            client = await self.get_client()
            await client.delete(self.PREFIX_RECENT)
            return True
        except Exception as e:
            logger.warning(f"Redis invalidate_recent error: {e}")
            return False

    # ============== Utility Methods ==============

    async def is_connected(self) -> bool:
        """Check if Redis is connected"""
        try:
            client = await self.get_client()
            await client.ping()
            return True
        except Exception:
            return False

    async def clear_all_memory_cache(self) -> int:
        """Clear all memory-related cache keys"""
        try:
            client = await self.get_client()

            # Find and delete all memory-related keys
            pattern = "memory:*"
            cursor = 0
            deleted = 0

            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=100)
                if keys:
                    await client.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break

            logger.info(f"Cleared {deleted} memory cache keys")
            return deleted

        except Exception as e:
            logger.warning(f"Redis clear_all_memory_cache error: {e}")
            return 0

    async def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        try:
            client = await self.get_client()

            # Count keys by prefix
            hash_keys = len(await client.keys(f"{self.PREFIX_QUERY_HASH}*"))
            result_keys = len(await client.keys(f"{self.PREFIX_QUERY_RESULT}*"))
            has_recent = await client.exists(self.PREFIX_RECENT)

            return {
                "connected": True,
                "hash_cache_count": hash_keys,
                "result_cache_count": result_keys,
                "has_recent_cache": bool(has_recent),
            }

        except Exception as e:
            return {
                "connected": False,
                "error": str(e)
            }


# Global service instance
_redis_cache: Optional[RedisCacheService] = None


async def get_redis_cache() -> RedisCacheService:
    """Get or create Redis cache service instance"""
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisCacheService()
    return _redis_cache
