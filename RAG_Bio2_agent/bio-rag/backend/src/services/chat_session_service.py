"""Chat Session Database Service - CRUD operations for chat sessions and messages"""

import logging
from typing import Optional, List
from uuid import UUID

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.chat import ChatSession, ChatMessage

logger = logging.getLogger(__name__)


class ChatSessionService:
    """Service for chat session database operations"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, session_id: UUID) -> Optional[ChatSession]:
        """Get session by ID"""
        result = await self.db.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        return result.scalar_one_or_none()

    async def get_by_id_with_messages(self, session_id: UUID) -> Optional[ChatSession]:
        """Get session by ID with messages loaded"""
        result = await self.db.execute(
            select(ChatSession)
            .where(ChatSession.id == session_id)
            .options(selectinload(ChatSession.messages))
        )
        return result.scalar_one_or_none()

    async def create(
        self,
        user_id: UUID,
        title: str = "New Conversation",
    ) -> ChatSession:
        """Create a new chat session"""
        session = ChatSession(
            user_id=user_id,
            title=title,
        )
        self.db.add(session)
        await self.db.flush()
        await self.db.refresh(session)
        logger.info(f"Created chat session: {session.id}")
        return session

    async def update_title(self, session_id: UUID, title: str) -> Optional[ChatSession]:
        """Update session title"""
        session = await self.get_by_id(session_id)
        if session:
            session.title = title
            await self.db.flush()
            await self.db.refresh(session)
        return session

    async def delete(self, session_id: UUID) -> bool:
        """Delete session by ID (cascades to messages)"""
        result = await self.db.execute(
            delete(ChatSession).where(ChatSession.id == session_id)
        )
        deleted = result.rowcount > 0
        if deleted:
            logger.info(f"Deleted chat session: {session_id}")
        return deleted

    async def list_by_user(
        self,
        user_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ChatSession]:
        """List all sessions for a user"""
        result = await self.db.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def add_message(
        self,
        session_id: UUID,
        role: str,
        content: str,
        sources: Optional[List[dict]] = None,
        confidence: Optional[float] = None,
        processing_time_ms: Optional[int] = None,
    ) -> ChatMessage:
        """Add a message to a session"""
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            sources=sources or [],
            confidence=str(confidence) if confidence else None,
            processing_time_ms=str(processing_time_ms) if processing_time_ms else None,
        )
        self.db.add(message)
        await self.db.flush()
        await self.db.refresh(message)
        return message

    async def get_messages(
        self,
        session_id: UUID,
        limit: int = 100,
    ) -> List[ChatMessage]:
        """Get all messages for a session"""
        result = await self.db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_message_count(self, session_id: UUID) -> int:
        """Get message count for a session"""
        from sqlalchemy import func
        result = await self.db.execute(
            select(func.count(ChatMessage.id))
            .where(ChatMessage.session_id == session_id)
        )
        return result.scalar() or 0
