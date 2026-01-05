"""Tests for Chat Memory Service with Redis + PostgreSQL"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.services.chat_memory import (
    ChatMemoryService,
    ConversationMemory,
    format_memory_context,
)
from src.services.redis_cache import RedisCacheService
from src.services.memory_store import MemoryStoreService


class TestConversationMemory:
    """Tests for ConversationMemory dataclass"""

    def test_create_conversation_memory(self):
        """Test creating a ConversationMemory object"""
        memory = ConversationMemory(
            id="test-id",
            query="What is CRISPR?",
            answer="CRISPR is a gene editing technology.",
            query_hash="abc123",
            created_at="2024-01-01T00:00:00",
            relevance_score=0.95,
        )

        assert memory.id == "test-id"
        assert memory.query == "What is CRISPR?"
        assert memory.relevance_score == 0.95

    def test_default_values(self):
        """Test default values"""
        memory = ConversationMemory(
            id="test-id",
            query="test",
            answer="test",
            query_hash="hash",
            created_at="2024-01-01",
        )

        assert memory.relevance_score == 0.0
        assert memory.sources_used is None


class TestChatMemoryService:
    """Tests for ChatMemoryService"""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session"""
        return AsyncMock()

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis cache service"""
        redis = AsyncMock(spec=RedisCacheService)
        redis.is_connected.return_value = True
        return redis

    @pytest.fixture
    def memory_service(self, mock_db, mock_redis):
        """Create a ChatMemoryService with mocks"""
        service = ChatMemoryService(db=mock_db, redis_cache=mock_redis)
        return service

    def test_hash_query(self, memory_service):
        """Test query hashing"""
        hash1 = memory_service._hash_query("What is CRISPR?")
        hash2 = memory_service._hash_query("what is crispr?")  # Different case
        hash3 = memory_service._hash_query("What is cancer?")

        # Same query (case insensitive) should produce same hash
        assert hash1 == hash2
        # Different query should produce different hash
        assert hash1 != hash3

    @pytest.mark.asyncio
    async def test_find_exact_match_cache_hit(self, memory_service, mock_redis):
        """Test finding exact match from Redis cache"""
        cached_data = {
            "id": "test-id",
            "query": "What is CRISPR?",
            "answer": "CRISPR is...",
            "query_hash": "abc123",
            "created_at": "2024-01-01T00:00:00",
            "relevance_score": 1.0,
        }
        mock_redis.get_by_hash.return_value = cached_data

        result = await memory_service.find_exact_match("What is CRISPR?")

        assert result is not None
        assert result.query == "What is CRISPR?"
        mock_redis.get_by_hash.assert_called_once()

    @pytest.mark.asyncio
    async def test_find_exact_match_cache_miss_db_hit(self, memory_service, mock_redis, mock_db):
        """Test finding exact match from PostgreSQL when cache misses"""
        mock_redis.get_by_hash.return_value = None

        # Mock the memory store
        mock_memory = MagicMock()
        mock_memory.id = uuid4()
        mock_memory.query = "What is CRISPR?"
        mock_memory.answer = "CRISPR is..."
        mock_memory.query_hash = "abc123"
        mock_memory.created_at = MagicMock()
        mock_memory.created_at.isoformat.return_value = "2024-01-01T00:00:00"
        mock_memory.relevance_score = 1.0
        mock_memory.sources_used = None
        mock_memory.to_dict.return_value = {
            "id": str(mock_memory.id),
            "query": mock_memory.query,
            "answer": mock_memory.answer,
            "query_hash": mock_memory.query_hash,
            "created_at": "2024-01-01T00:00:00",
        }

        with patch.object(memory_service.store, 'find_by_hash', return_value=mock_memory):
            result = await memory_service.find_exact_match("What is CRISPR?")

        assert result is not None
        assert result.query == "What is CRISPR?"
        # Should cache the result
        mock_redis.set_by_hash.assert_called_once()

    @pytest.mark.asyncio
    async def test_find_exact_match_not_found(self, memory_service, mock_redis):
        """Test when no exact match is found"""
        mock_redis.get_by_hash.return_value = None

        with patch.object(memory_service.store, 'find_by_hash', return_value=None):
            result = await memory_service.find_exact_match("Unknown query")

        assert result is None

    @pytest.mark.asyncio
    async def test_save_conversation(self, memory_service, mock_redis):
        """Test saving a conversation"""
        mock_memory = MagicMock()
        mock_memory.id = uuid4()
        mock_memory.query_hash = "abc123"
        mock_memory.to_dict.return_value = {"id": str(mock_memory.id)}

        with patch.object(memory_service.store, 'save', return_value=mock_memory):
            result = await memory_service.save_conversation(
                query="What is CRISPR?",
                answer="CRISPR is...",
                sources_used=["12345", "67890"]
            )

        assert result == str(mock_memory.id)
        # Should cache in Redis
        mock_redis.set_by_hash.assert_called_once()
        # Should invalidate recent cache
        mock_redis.invalidate_recent.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_fallback(self, mock_db):
        """Test fallback to PostgreSQL when Redis is unavailable"""
        # Create service with failing Redis
        mock_redis = AsyncMock(spec=RedisCacheService)
        mock_redis.is_connected.return_value = False

        service = ChatMemoryService(db=mock_db, redis_cache=mock_redis)

        # Should handle Redis failure gracefully
        with patch.object(service.store, 'find_by_hash', return_value=None):
            result = await service.find_exact_match("Test query")

        assert result is None
        assert service._redis_available is False

    @pytest.mark.asyncio
    async def test_get_cache_stats(self, memory_service, mock_redis):
        """Test getting cache statistics"""
        mock_redis.get_cache_stats.return_value = {
            "connected": True,
            "hash_cache_count": 10,
            "result_cache_count": 5,
        }

        with patch.object(memory_service.store, 'get_count', return_value=100):
            stats = await memory_service.get_cache_stats()

        assert stats["postgresql_count"] == 100
        assert stats["redis_available"] is True
        assert "redis" in stats


class TestFormatMemoryContext:
    """Tests for format_memory_context function"""

    def test_empty_memories(self):
        """Test with empty memories list"""
        result = format_memory_context([])
        assert result == ""

    def test_single_memory(self):
        """Test with single memory"""
        memories = [
            ConversationMemory(
                id="1",
                query="What is CRISPR?",
                answer="CRISPR is a gene editing technology.",
                query_hash="hash1",
                created_at="2024-01-01",
                relevance_score=0.95,
            )
        ]

        result = format_memory_context(memories)

        assert "What is CRISPR?" in result
        assert "CRISPR is a gene editing technology." in result
        assert "0.95" in result

    def test_multiple_memories(self):
        """Test with multiple memories"""
        memories = [
            ConversationMemory(
                id="1",
                query="Query 1",
                answer="Answer 1",
                query_hash="hash1",
                created_at="2024-01-01",
                relevance_score=0.9,
            ),
            ConversationMemory(
                id="2",
                query="Query 2",
                answer="Answer 2",
                query_hash="hash2",
                created_at="2024-01-02",
                relevance_score=0.8,
            ),
        ]

        result = format_memory_context(memories)

        assert "Query 1" in result
        assert "Query 2" in result
        assert "과거 질문 1" in result
        assert "과거 질문 2" in result

    def test_long_answer_truncation(self):
        """Test that long answers are truncated"""
        long_answer = "A" * 600  # Longer than 500 chars
        memories = [
            ConversationMemory(
                id="1",
                query="Query",
                answer=long_answer,
                query_hash="hash1",
                created_at="2024-01-01",
                relevance_score=0.9,
            )
        ]

        result = format_memory_context(memories)

        # Should be truncated with "..."
        assert "..." in result
        # Should not contain full answer
        assert long_answer not in result


class TestRedisCacheService:
    """Tests for RedisCacheService"""

    def test_cache_key_prefixes(self):
        """Test cache key prefix constants"""
        assert RedisCacheService.PREFIX_QUERY_HASH == "memory:hash:"
        assert RedisCacheService.PREFIX_QUERY_RESULT == "memory:result:"
        assert RedisCacheService.PREFIX_RECENT == "memory:recent"

    def test_ttl_values(self):
        """Test TTL value constants"""
        assert RedisCacheService.TTL_QUERY_HASH == 3600 * 24  # 24 hours
        assert RedisCacheService.TTL_QUERY_RESULT == 3600  # 1 hour
        assert RedisCacheService.TTL_RECENT == 3600 * 6  # 6 hours

    @pytest.mark.asyncio
    async def test_is_connected_failure(self):
        """Test is_connected when Redis is unavailable"""
        with patch('redis.asyncio.from_url') as mock_from_url:
            mock_client = AsyncMock()
            mock_client.ping.side_effect = Exception("Connection failed")
            mock_from_url.return_value = mock_client

            service = RedisCacheService()
            service._client = mock_client

            result = await service.is_connected()

            assert result is False
