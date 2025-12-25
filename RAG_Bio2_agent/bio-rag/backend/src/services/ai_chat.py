"""AI Chat Service - OpenAI/Anthropic Integration with RAG"""

import logging
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import aiohttp

from src.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ChatSource:
    """Source document used for RAG"""
    pmid: str
    title: str
    abstract: str
    relevance: float


@dataclass
class ChatResponse:
    """AI chat response"""
    answer: str
    sources_used: List[str]  # PMIDs
    confidence: float


class AIService:
    """AI Service for chat with RAG support"""

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        provider: str = "openai"  # "openai" or "anthropic"
    ):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model
        self.provider = provider
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the session"""
        if self._session and not self._session.closed:
            await self._session.close()

    def _build_system_prompt(self) -> str:
        """Build system prompt for biomedical RAG"""
        return """You are Bio-RAG, an AI research assistant specialized in biomedical literature analysis.

Your role is to:
1. Answer questions about biomedical research based on provided paper contexts
2. Cite sources using PMID references when providing information
3. Be accurate and scientific in your responses
4. Acknowledge limitations when information is incomplete

Guidelines:
- Always reference the source papers using their PMID (e.g., "According to PMID:12345..." or "PMID:12345에 따르면...")
- If the provided papers don't contain enough information, say so honestly
- Use scientific terminology appropriately
- Provide concise but comprehensive answers
- When discussing research findings, mention key details like sample sizes, methods, and results

Language:
- IMPORTANT: Respond in the SAME LANGUAGE as the user's question
- If the user asks in Korean (한국어), respond entirely in Korean
- If the user asks in English, respond in English
- Use appropriate scientific terminology in the target language

Response format:
- Start with a direct answer to the question
- Support claims with citations from the provided papers
- End with any relevant caveats or suggestions for further reading"""

    def _build_context_prompt(self, question: str, sources: List[ChatSource]) -> str:
        """Build context-aware prompt with paper information"""
        context_parts = ["Here are relevant research papers to help answer the question:\n"]

        for i, source in enumerate(sources, 1):
            context_parts.append(f"""
Paper {i}:
- PMID: {source.pmid}
- Title: {source.title}
- Abstract: {source.abstract}
- Relevance Score: {source.relevance:.2f}
""")

        context_parts.append(f"\nUser Question: {question}")
        context_parts.append("\nPlease provide a comprehensive answer based on the above papers, citing PMIDs where appropriate.")

        return "".join(context_parts)

    async def chat_with_context(
        self,
        question: str,
        sources: List[ChatSource],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> ChatResponse:
        """
        Generate AI response with RAG context

        Args:
            question: User's question
            sources: List of relevant papers as context
            conversation_history: Optional previous messages

        Returns:
            ChatResponse with answer and metadata
        """
        if not self.api_key:
            logger.warning("No API key configured, using fallback response")
            return self._fallback_response(question, sources)

        try:
            if self.provider == "openai":
                return await self._openai_chat(question, sources, conversation_history)
            elif self.provider == "anthropic":
                return await self._anthropic_chat(question, sources, conversation_history)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

        except Exception as e:
            logger.error(f"AI chat error: {e}")
            return self._fallback_response(question, sources)

    async def _openai_chat(
        self,
        question: str,
        sources: List[ChatSource],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> ChatResponse:
        """Call OpenAI API"""
        session = await self._get_session()

        messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ]

        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history[-6:])  # Last 6 messages for context

        # Add current question with paper context
        user_message = self._build_context_prompt(question, sources)
        messages.append({"role": "user", "content": user_message})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1500
        }

        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"OpenAI API error: {response.status} - {error_text}")
                return self._fallback_response(question, sources)

            data = await response.json()
            answer = data["choices"][0]["message"]["content"]

            # Extract which PMIDs were actually used in the response
            sources_used = [s.pmid for s in sources if s.pmid in answer]

            # Calculate confidence based on sources used and response quality
            confidence = min(0.95, 0.5 + (len(sources_used) * 0.1))

            return ChatResponse(
                answer=answer,
                sources_used=sources_used,
                confidence=confidence
            )

    async def _anthropic_chat(
        self,
        question: str,
        sources: List[ChatSource],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> ChatResponse:
        """Call Anthropic Claude API"""
        session = await self._get_session()

        # Build messages for Claude
        messages = []

        if conversation_history:
            for msg in conversation_history[-6:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        user_message = self._build_context_prompt(question, sources)
        messages.append({"role": "user", "content": user_message})

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": "claude-3-haiku-20240307",  # Fast and cost-effective
            "max_tokens": 1500,
            "system": self._build_system_prompt(),
            "messages": messages
        }

        async with session.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Anthropic API error: {response.status} - {error_text}")
                return self._fallback_response(question, sources)

            data = await response.json()
            answer = data["content"][0]["text"]

            sources_used = [s.pmid for s in sources if s.pmid in answer]
            confidence = min(0.95, 0.5 + (len(sources_used) * 0.1))

            return ChatResponse(
                answer=answer,
                sources_used=sources_used,
                confidence=confidence
            )

    def _fallback_response(self, question: str, sources: List[ChatSource]) -> ChatResponse:
        """Generate fallback response when AI is unavailable"""
        if not sources:
            return ChatResponse(
                answer="I couldn't find relevant papers for your question. Please try rephrasing or use different keywords.",
                sources_used=[],
                confidence=0.1
            )

        # Build a basic response from the sources
        answer_parts = [
            f"Based on my search, I found {len(sources)} relevant papers for your question about '{question[:50]}...':\n\n"
        ]

        for i, source in enumerate(sources[:3], 1):
            answer_parts.append(
                f"**{i}. {source.title}** (PMID: {source.pmid})\n"
                f"{source.abstract[:300]}...\n\n"
            )

        answer_parts.append(
            "\n*Note: AI-powered analysis is currently unavailable. "
            "Please refer to the full papers for detailed information.*"
        )

        return ChatResponse(
            answer="".join(answer_parts),
            sources_used=[s.pmid for s in sources[:3]],
            confidence=0.3
        )


# Global service instance
_ai_service: Optional[AIService] = None


def get_ai_service() -> AIService:
    """Get or create AI service instance"""
    global _ai_service
    if _ai_service is None:
        _ai_service = AIService(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL if hasattr(settings, 'OPENAI_MODEL') else "gpt-4o-mini"
        )
    return _ai_service
