"""Chat Memory Model for PostgreSQL"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, Float, Index
from sqlalchemy.dialects.postgresql import UUID

from src.core.database import Base


class ChatMemory(Base):
    """Chat conversation memory for RAG enhancement"""

    __tablename__ = "chat_memories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    query_hash = Column(String(64), nullable=False, index=True)
    sources_used = Column(Text, nullable=True)  # Comma-separated PMIDs
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)  # Optional user association
    relevance_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Composite index for faster lookups
    __table_args__ = (
        Index('idx_query_hash_created', 'query_hash', 'created_at'),
    )

    def __repr__(self):
        return f"<ChatMemory {self.id}: {self.query[:50]}...>"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "id": str(self.id),
            "query": self.query,
            "answer": self.answer,
            "query_hash": self.query_hash,
            "sources_used": self.sources_used,
            "relevance_score": self.relevance_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
