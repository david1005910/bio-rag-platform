"""Tests for AI Chat Service"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.ai_chat import AIService, ChatSource, ChatResponse, get_ai_service


class AsyncContextManager:
    """Helper class to mock async context managers"""

    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


# Sample data for testing
SAMPLE_SOURCES = [
    ChatSource(
        pmid="12345678",
        title="CRISPR Gene Therapy for Cancer",
        abstract="This study explores CRISPR-based approaches for cancer treatment...",
        relevance=0.95
    ),
    ChatSource(
        pmid="87654321",
        title="Immunotherapy Advances in Oncology",
        abstract="Recent advances in immunotherapy have revolutionized cancer treatment...",
        relevance=0.88
    )
]


class TestChatSource:
    """Test ChatSource dataclass"""

    def test_create_chat_source(self):
        """Test creating ChatSource instance"""
        source = ChatSource(
            pmid="12345678",
            title="Test Paper",
            abstract="Test abstract content",
            relevance=0.95
        )
        assert source.pmid == "12345678"
        assert source.title == "Test Paper"
        assert source.relevance == 0.95

    def test_chat_source_relevance_range(self):
        """Test ChatSource with various relevance values"""
        source_high = ChatSource("1", "High", "Abstract", 1.0)
        source_low = ChatSource("2", "Low", "Abstract", 0.1)

        assert source_high.relevance == 1.0
        assert source_low.relevance == 0.1


class TestChatResponse:
    """Test ChatResponse dataclass"""

    def test_create_chat_response(self):
        """Test creating ChatResponse instance"""
        response = ChatResponse(
            answer="This is the AI answer",
            sources_used=["12345678", "87654321"],
            confidence=0.85
        )
        assert response.answer == "This is the AI answer"
        assert len(response.sources_used) == 2
        assert response.confidence == 0.85

    def test_chat_response_empty_sources(self):
        """Test ChatResponse with no sources"""
        response = ChatResponse(
            answer="No sources found",
            sources_used=[],
            confidence=0.1
        )
        assert response.sources_used == []
        assert response.confidence == 0.1


class TestAIService:
    """Test AIService class"""

    @pytest.fixture
    def ai_service(self):
        """Create AIService instance for testing"""
        return AIService(api_key="test_api_key", model="gpt-4o-mini", provider="openai")

    @pytest.fixture
    def ai_service_no_key(self):
        """Create AIService instance without API key"""
        with patch('src.services.ai_chat.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = ""
            return AIService(api_key="", model="gpt-4o-mini")

    def test_init_with_api_key(self):
        """Test initialization with API key"""
        service = AIService(api_key="my_key", model="gpt-4", provider="openai")
        assert service.api_key == "my_key"
        assert service.model == "gpt-4"
        assert service.provider == "openai"

    def test_init_with_anthropic(self):
        """Test initialization with Anthropic provider"""
        service = AIService(api_key="anthropic_key", provider="anthropic")
        assert service.provider == "anthropic"

    def test_build_system_prompt(self, ai_service):
        """Test system prompt generation"""
        prompt = ai_service._build_system_prompt()

        assert "Bio-RAG" in prompt
        assert "PMID" in prompt
        assert "Korean" in prompt or "한국어" in prompt
        assert len(prompt) > 500  # Should be a comprehensive prompt

    def test_build_context_prompt_english(self, ai_service):
        """Test context prompt with English question"""
        sources = SAMPLE_SOURCES
        question = "What are the latest advances in CRISPR therapy?"

        prompt = ai_service._build_context_prompt(question, sources)

        assert "PMID: 12345678" in prompt
        assert "PMID: 87654321" in prompt
        assert "CRISPR Gene Therapy" in prompt
        assert question in prompt
        assert "English" in prompt  # Language instruction

    def test_build_context_prompt_korean(self, ai_service):
        """Test context prompt with Korean question"""
        sources = SAMPLE_SOURCES
        question = "암 치료에 대한 최신 연구는 무엇인가요?"

        prompt = ai_service._build_context_prompt(question, sources)

        assert "PMID: 12345678" in prompt
        assert question in prompt
        assert "한국어" in prompt  # Korean language instruction

    def test_build_context_prompt_empty_sources(self, ai_service):
        """Test context prompt with no sources"""
        prompt = ai_service._build_context_prompt("Test question?", [])

        assert "Test question?" in prompt
        assert "Paper 1" not in prompt

    @pytest.mark.asyncio
    async def test_chat_with_context_no_api_key(self, ai_service_no_key):
        """Test chat returns fallback when no API key"""
        response = await ai_service_no_key.chat_with_context(
            question="Test question?",
            sources=SAMPLE_SOURCES
        )

        assert isinstance(response, ChatResponse)
        assert response.confidence < 0.5  # Low confidence for fallback

    @pytest.mark.asyncio
    async def test_openai_chat_success(self, ai_service):
        """Test successful OpenAI API call"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{
                "message": {
                    "content": "Based on PMID:12345678, CRISPR therapy shows promise..."
                }
            }]
        })

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncContextManager(mock_response))

        with patch.object(ai_service, '_get_session', return_value=mock_session):
            response = await ai_service._openai_chat(
                question="What is CRISPR?",
                sources=SAMPLE_SOURCES
            )

        assert isinstance(response, ChatResponse)
        assert "CRISPR" in response.answer
        assert "12345678" in response.sources_used

    @pytest.mark.asyncio
    async def test_openai_chat_api_error(self, ai_service):
        """Test OpenAI API error handling"""
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncContextManager(mock_response))

        with patch.object(ai_service, '_get_session', return_value=mock_session):
            response = await ai_service._openai_chat(
                question="Test question?",
                sources=SAMPLE_SOURCES
            )

        # Should return fallback response
        assert isinstance(response, ChatResponse)
        assert response.confidence < 0.5

    @pytest.mark.asyncio
    async def test_openai_chat_with_history(self, ai_service):
        """Test OpenAI chat with conversation history"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{
                "message": {
                    "content": "Following up on our discussion about PMID:12345678..."
                }
            }]
        })

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncContextManager(mock_response))

        history = [
            {"role": "user", "content": "Tell me about cancer treatment"},
            {"role": "assistant", "content": "Cancer treatment has many approaches..."}
        ]

        with patch.object(ai_service, '_get_session', return_value=mock_session):
            response = await ai_service._openai_chat(
                question="What about CRISPR specifically?",
                sources=SAMPLE_SOURCES,
                conversation_history=history
            )

        assert isinstance(response, ChatResponse)

    @pytest.mark.asyncio
    async def test_anthropic_chat_success(self):
        """Test successful Anthropic API call"""
        service = AIService(api_key="anthropic_key", provider="anthropic")

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "content": [{
                "text": "Based on PMID:12345678, the research shows..."
            }]
        })

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncContextManager(mock_response))

        with patch.object(service, '_get_session', return_value=mock_session):
            response = await service._anthropic_chat(
                question="Explain CRISPR therapy",
                sources=SAMPLE_SOURCES
            )

        assert isinstance(response, ChatResponse)
        assert "12345678" in response.sources_used

    @pytest.mark.asyncio
    async def test_anthropic_chat_api_error(self):
        """Test Anthropic API error handling"""
        service = AIService(api_key="anthropic_key", provider="anthropic")

        mock_response = MagicMock()
        mock_response.status = 401
        mock_response.text = AsyncMock(return_value="Unauthorized")

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncContextManager(mock_response))

        with patch.object(service, '_get_session', return_value=mock_session):
            response = await service._anthropic_chat(
                question="Test?",
                sources=SAMPLE_SOURCES
            )

        assert response.confidence < 0.5

    @pytest.mark.asyncio
    async def test_chat_with_context_openai_provider(self, ai_service):
        """Test chat_with_context routes to OpenAI"""
        mock_response = ChatResponse(
            answer="Test answer",
            sources_used=["12345678"],
            confidence=0.8
        )

        with patch.object(ai_service, '_openai_chat', return_value=mock_response) as mock_openai:
            response = await ai_service.chat_with_context(
                question="Test?",
                sources=SAMPLE_SOURCES
            )

            mock_openai.assert_called_once()
            assert response.answer == "Test answer"

    @pytest.mark.asyncio
    async def test_chat_with_context_anthropic_provider(self):
        """Test chat_with_context routes to Anthropic"""
        service = AIService(api_key="key", provider="anthropic")

        mock_response = ChatResponse(
            answer="Anthropic answer",
            sources_used=["12345678"],
            confidence=0.85
        )

        with patch.object(service, '_anthropic_chat', return_value=mock_response) as mock_anthropic:
            response = await service.chat_with_context(
                question="Test?",
                sources=SAMPLE_SOURCES
            )

            mock_anthropic.assert_called_once()
            assert response.answer == "Anthropic answer"

    @pytest.mark.asyncio
    async def test_chat_with_context_unknown_provider(self):
        """Test chat_with_context with unknown provider"""
        service = AIService(api_key="key", provider="unknown_provider")

        response = await service.chat_with_context(
            question="Test?",
            sources=SAMPLE_SOURCES
        )

        # Should return fallback due to exception
        assert isinstance(response, ChatResponse)

    @pytest.mark.asyncio
    async def test_chat_with_context_exception_handling(self, ai_service):
        """Test chat_with_context handles exceptions gracefully"""
        with patch.object(ai_service, '_openai_chat', side_effect=Exception("Network error")):
            response = await ai_service.chat_with_context(
                question="Test?",
                sources=SAMPLE_SOURCES
            )

        # Should return fallback response
        assert isinstance(response, ChatResponse)
        assert response.confidence < 0.5

    def test_fallback_response_with_sources(self, ai_service):
        """Test fallback response when sources are available"""
        response = ai_service._fallback_response(
            question="What is CRISPR therapy?",
            sources=SAMPLE_SOURCES
        )

        assert "CRISPR Gene Therapy" in response.answer
        assert "12345678" in response.answer
        assert len(response.sources_used) > 0
        assert response.confidence == 0.3

    def test_fallback_response_no_sources(self, ai_service):
        """Test fallback response when no sources"""
        response = ai_service._fallback_response(
            question="Random question?",
            sources=[]
        )

        assert "couldn't find" in response.answer.lower()
        assert response.sources_used == []
        assert response.confidence == 0.1

    def test_fallback_response_truncates_abstract(self, ai_service):
        """Test fallback truncates long abstracts"""
        long_abstract = "A" * 500
        sources = [ChatSource(
            pmid="11111111",
            title="Long Paper",
            abstract=long_abstract,
            relevance=0.9
        )]

        response = ai_service._fallback_response("Test?", sources)

        # Abstract should be truncated to ~300 chars + "..."
        assert "..." in response.answer

    @pytest.mark.asyncio
    async def test_close_session(self, ai_service):
        """Test closing the session"""
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()

        ai_service._session = mock_session
        await ai_service.close()

        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_session_already_closed(self, ai_service):
        """Test closing already closed session"""
        mock_session = MagicMock()
        mock_session.closed = True

        ai_service._session = mock_session
        await ai_service.close()  # Should not raise


class TestConfidenceCalculation:
    """Test confidence score calculation"""

    @pytest.fixture
    def ai_service(self):
        return AIService(api_key="test_key")

    @pytest.mark.asyncio
    async def test_confidence_increases_with_sources(self, ai_service):
        """Test that confidence increases with more sources used"""
        # Mock response mentioning all source PMIDs
        all_pmids_answer = "Based on PMID:12345678 and PMID:87654321, the research shows..."

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": all_pmids_answer}}]
        })

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncContextManager(mock_response))

        with patch.object(ai_service, '_get_session', return_value=mock_session):
            response = await ai_service._openai_chat("Test?", SAMPLE_SOURCES)

        # Should have higher confidence with 2 sources
        assert len(response.sources_used) == 2
        assert response.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_confidence_capped_at_095(self, ai_service):
        """Test that confidence is capped at 0.95"""
        # Create many sources
        many_sources = [
            ChatSource(pmid=str(i), title=f"Paper {i}", abstract="Test", relevance=0.9)
            for i in range(10)
        ]

        # Mock response mentioning all PMIDs
        all_pmids = " ".join([f"PMID:{s.pmid}" for s in many_sources])
        answer = f"Based on {all_pmids}, the research is conclusive."

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": answer}}]
        })

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncContextManager(mock_response))

        with patch.object(ai_service, '_get_session', return_value=mock_session):
            response = await ai_service._openai_chat("Test?", many_sources)

        # Confidence should be capped
        assert response.confidence == 0.95


class TestGetAIService:
    """Test get_ai_service function"""

    def test_get_ai_service_singleton(self):
        """Test that get_ai_service returns singleton instance"""
        # Reset the global instance
        import src.services.ai_chat as ai_chat_module
        ai_chat_module._ai_service = None

        with patch.object(ai_chat_module.settings, 'OPENAI_API_KEY', 'test_key'):
            service1 = get_ai_service()
            service2 = get_ai_service()

            assert service1 is service2
            assert isinstance(service1, AIService)
