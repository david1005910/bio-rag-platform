"""Chat Memory Service - SQLite-based conversation history for RAG enhancement"""

import sqlite3
import logging
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent.parent / "data" / "chat_memory.db"


@dataclass
class ConversationMemory:
    """Stored conversation memory"""
    id: int
    query: str
    answer: str
    query_hash: str
    created_at: str
    relevance_score: float = 0.0


class ChatMemoryService:
    """SQLite-based chat memory for storing and retrieving past Q&A"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self._ensure_db_exists()
        self._init_database()

    def _ensure_db_exists(self):
        """Ensure the database directory exists"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        """Initialize the database schema"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Create conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    query_hash TEXT NOT NULL,
                    sources_used TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create FTS virtual table for full-text search
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
                    query,
                    answer,
                    content='conversations',
                    content_rowid='id'
                )
            """)

            # Create triggers to keep FTS in sync
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS conversations_ai AFTER INSERT ON conversations BEGIN
                    INSERT INTO conversations_fts(rowid, query, answer)
                    VALUES (new.id, new.query, new.answer);
                END
            """)

            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS conversations_ad AFTER DELETE ON conversations BEGIN
                    INSERT INTO conversations_fts(conversations_fts, rowid, query, answer)
                    VALUES('delete', old.id, old.query, old.answer);
                END
            """)

            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS conversations_au AFTER UPDATE ON conversations BEGIN
                    INSERT INTO conversations_fts(conversations_fts, rowid, query, answer)
                    VALUES('delete', old.id, old.query, old.answer);
                    INSERT INTO conversations_fts(rowid, query, answer)
                    VALUES (new.id, new.query, new.answer);
                END
            """)

            # Create index for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_hash ON conversations(query_hash)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON conversations(created_at DESC)
            """)

            conn.commit()
            logger.info(f"Chat memory database initialized at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
        finally:
            conn.close()

    def _hash_query(self, query: str) -> str:
        """Generate a hash for the query for exact match detection"""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def save_conversation(
        self,
        query: str,
        answer: str,
        sources_used: Optional[List[str]] = None
    ) -> int:
        """
        Save a conversation to the database

        Args:
            query: User's question
            answer: AI's response
            sources_used: List of PMIDs used in the response

        Returns:
            The ID of the saved conversation
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            query_hash = self._hash_query(query)
            sources_str = ",".join(sources_used) if sources_used else ""

            cursor.execute("""
                INSERT INTO conversations (query, answer, query_hash, sources_used)
                VALUES (?, ?, ?, ?)
            """, (query, answer, query_hash, sources_str))

            conn.commit()
            conversation_id = cursor.lastrowid

            logger.info(f"Saved conversation {conversation_id}: '{query[:50]}...'")
            return conversation_id

        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            raise
        finally:
            conn.close()

    def find_exact_match(self, query: str) -> Optional[ConversationMemory]:
        """
        Find an exact match for a query (same question asked before)

        Args:
            query: User's question

        Returns:
            ConversationMemory if found, None otherwise
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            query_hash = self._hash_query(query)

            cursor.execute("""
                SELECT id, query, answer, query_hash, created_at
                FROM conversations
                WHERE query_hash = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (query_hash,))

            row = cursor.fetchone()
            if row:
                return ConversationMemory(
                    id=row['id'],
                    query=row['query'],
                    answer=row['answer'],
                    query_hash=row['query_hash'],
                    created_at=row['created_at'],
                    relevance_score=1.0
                )
            return None

        finally:
            conn.close()

    def search_similar_conversations(
        self,
        query: str,
        limit: int = 3,
        min_relevance: float = 0.1
    ) -> List[ConversationMemory]:
        """
        Search for similar past conversations using FTS

        Args:
            query: User's question
            limit: Maximum number of results
            min_relevance: Minimum relevance score (BM25-based)

        Returns:
            List of relevant ConversationMemory objects
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Extract keywords from the query for FTS search
            # Remove common Korean/English stop words
            stop_words = {
                '은', '는', '이', '가', '을', '를', '의', '에', '에서', '로', '으로',
                '와', '과', '도', '만', '에게', '한테', '께', '보다', '처럼', '같이',
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                'what', 'how', 'why', 'when', 'where', 'which', 'who',
                '무엇', '어떻게', '왜', '언제', '어디', '누가', '뭐', '뭘'
            }

            # Tokenize and filter
            words = query.replace('?', '').replace('!', '').replace('.', '').split()
            keywords = [w for w in words if w.lower() not in stop_words and len(w) > 1]

            if not keywords:
                return []

            # Build FTS query with OR between keywords
            fts_query = " OR ".join(keywords)

            cursor.execute("""
                SELECT
                    c.id,
                    c.query,
                    c.answer,
                    c.query_hash,
                    c.created_at,
                    bm25(conversations_fts) as relevance
                FROM conversations_fts
                JOIN conversations c ON conversations_fts.rowid = c.id
                WHERE conversations_fts MATCH ?
                ORDER BY relevance
                LIMIT ?
            """, (fts_query, limit * 2))  # Get more results to filter

            results = []
            for row in cursor.fetchall():
                # Normalize BM25 score (lower is better, so we invert)
                # BM25 scores are typically negative, closer to 0 is better
                relevance = max(0, 1 + row['relevance'] / 10)  # Normalize to 0-1ish

                if relevance >= min_relevance:
                    results.append(ConversationMemory(
                        id=row['id'],
                        query=row['query'],
                        answer=row['answer'],
                        query_hash=row['query_hash'],
                        created_at=row['created_at'],
                        relevance_score=relevance
                    ))

            # Sort by relevance and return top results
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"Failed to search conversations: {e}")
            return []
        finally:
            conn.close()

    def get_recent_conversations(self, limit: int = 5) -> List[ConversationMemory]:
        """
        Get the most recent conversations

        Args:
            limit: Maximum number of results

        Returns:
            List of recent ConversationMemory objects
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, query, answer, query_hash, created_at
                FROM conversations
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))

            results = []
            for row in cursor.fetchall():
                results.append(ConversationMemory(
                    id=row['id'],
                    query=row['query'],
                    answer=row['answer'],
                    query_hash=row['query_hash'],
                    created_at=row['created_at'],
                    relevance_score=0.5  # Default score for recent
                ))

            return results

        finally:
            conn.close()

    def get_conversation_count(self) -> int:
        """Get the total number of stored conversations"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM conversations")
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def clear_old_conversations(self, days: int = 30) -> int:
        """
        Clear conversations older than specified days

        Args:
            days: Number of days to keep

        Returns:
            Number of deleted conversations
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM conversations
                WHERE created_at < datetime('now', ? || ' days')
            """, (f"-{days}",))

            deleted_count = cursor.rowcount
            conn.commit()

            logger.info(f"Cleared {deleted_count} old conversations")
            return deleted_count

        finally:
            conn.close()


# Global service instance
_memory_service: Optional[ChatMemoryService] = None


def get_memory_service() -> ChatMemoryService:
    """Get or create chat memory service instance"""
    global _memory_service
    if _memory_service is None:
        _memory_service = ChatMemoryService()
    return _memory_service


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
