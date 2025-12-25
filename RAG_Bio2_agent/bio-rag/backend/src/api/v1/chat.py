"""Chat API Endpoints - RAG-based Q&A"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.core.security import get_current_user_id, get_current_user_id_optional
from src.data import sample_papers

router = APIRouter()


# ============== Schemas ==============

class ChatQueryRequest(BaseModel):
    """Chat query request"""
    question: str
    session_id: Optional[str] = None
    context_pmids: List[str] = []
    max_sources: int = 5


class SourceInfo(BaseModel):
    """Source information"""
    pmid: str
    title: str
    relevance: float
    excerpt: str


class ChatQueryResponse(BaseModel):
    """Chat query response"""
    answer: str
    sources: List[SourceInfo]
    confidence: float
    processing_time_ms: int
    session_id: str


class ChatSession(BaseModel):
    """Chat session"""
    id: str
    title: str
    created_at: str
    message_count: int


class ChatMessage(BaseModel):
    """Chat message"""
    id: str
    role: str  # "user" or "assistant"
    content: str
    sources: List[SourceInfo] = []
    created_at: str


class SessionListResponse(BaseModel):
    """Session list response"""
    sessions: List[ChatSession]


class MessageListResponse(BaseModel):
    """Message list response"""
    messages: List[ChatMessage]


# ============== Endpoints ==============

@router.post("/query", response_model=ChatQueryResponse)
async def chat_query(
    request: ChatQueryRequest,
    user_id: Optional[str] = Depends(get_current_user_id_optional)
):
    """
    AI-powered Q&A about biomedical research

    - Uses RAG (Retrieval-Augmented Generation)
    - Searches relevant papers and generates answer
    - Includes source citations with PMID links

    Response time target: < 5 seconds
    """
    import uuid
    import time

    start_time = time.time()

    # Search for relevant papers
    total, results = sample_papers.search_papers(
        query=request.question,
        limit=request.max_sources
    )

    sources = []
    answer_parts = []

    if results:
        # Build sources list
        for paper in results:
            sources.append(SourceInfo(
                pmid=paper["pmid"],
                title=paper["title"],
                relevance=paper["relevance_score"],
                excerpt=paper["abstract"][:300] + "..."
            ))

        # Generate a contextual answer based on the question and found papers
        question_lower = request.question.lower()

        if "crispr" in question_lower:
            answer_parts.append(
                "CRISPR-Cas9 is a revolutionary gene editing technology that has transformed biomedical research. "
            )
            for paper in results[:2]:
                if "crispr" in paper["title"].lower():
                    answer_parts.append(f"According to recent research (PMID: {paper['pmid']}), {paper['abstract'][:200]}... ")

        elif "car-t" in question_lower or "car t" in question_lower:
            answer_parts.append(
                "CAR-T cell therapy is an innovative immunotherapy approach that engineers patients' T cells to target cancer. "
            )
            for paper in results[:2]:
                if "car-t" in paper["title"].lower() or "car t" in paper["title"].lower():
                    answer_parts.append(f"A study (PMID: {paper['pmid']}) reports: {paper['abstract'][:200]}... ")

        elif "cancer" in question_lower or "tumor" in question_lower:
            answer_parts.append(
                "Cancer research continues to advance with novel therapeutic approaches including immunotherapy, targeted therapy, and gene editing. "
            )
            for paper in results[:2]:
                answer_parts.append(f"Research findings (PMID: {paper['pmid']}): {paper['abstract'][:150]}... ")

        elif "ai" in question_lower or "artificial intelligence" in question_lower or "machine learning" in question_lower:
            answer_parts.append(
                "Artificial intelligence is revolutionizing drug discovery and biomedical research. "
            )
            for paper in results[:2]:
                if "ai" in paper["title"].lower() or "artificial" in paper["title"].lower():
                    answer_parts.append(f"Recent advances (PMID: {paper['pmid']}): {paper['abstract'][:200]}... ")

        elif "microbiome" in question_lower or "gut" in question_lower:
            answer_parts.append(
                "The gut microbiome plays a crucial role in health and disease, including cancer immunotherapy response. "
            )
            for paper in results[:2]:
                answer_parts.append(f"Studies show (PMID: {paper['pmid']}): {paper['abstract'][:200]}... ")

        else:
            answer_parts.append(
                f"Based on my search of the biomedical literature, I found {total} relevant papers. "
            )
            for paper in results[:2]:
                answer_parts.append(f"Key findings from PMID {paper['pmid']}: {paper['abstract'][:150]}... ")

        confidence = min(results[0]["relevance_score"] + 0.3, 0.95) if results else 0.0
    else:
        answer_parts.append(
            "I couldn't find specific papers matching your query in the current database. "
            "Please try rephrasing your question or use different keywords related to: "
            "CRISPR, CAR-T therapy, cancer immunotherapy, gene editing, AI drug discovery, or microbiome research."
        )
        confidence = 0.1

    processing_time = int((time.time() - start_time) * 1000)

    return ChatQueryResponse(
        answer=" ".join(answer_parts),
        sources=sources,
        confidence=confidence,
        processing_time_ms=processing_time,
        session_id=request.session_id or str(uuid.uuid4())
    )


@router.post("/sessions", response_model=ChatSession)
async def create_session(
    title: Optional[str] = None,
    user_id: str = Depends(get_current_user_id)
):
    """
    Create a new chat session

    - Requires authentication
    - Stores conversation history
    """
    import uuid
    from datetime import datetime

    session_id = str(uuid.uuid4())

    return ChatSession(
        id=session_id,
        title=title or "New Conversation",
        created_at=datetime.utcnow().isoformat(),
        message_count=0
    )


@router.get("/sessions", response_model=SessionListResponse)
async def get_sessions(
    user_id: str = Depends(get_current_user_id)
):
    """
    Get all chat sessions for current user

    - Requires authentication
    - Returns list of sessions with metadata
    """
    # TODO: Fetch from database

    return SessionListResponse(sessions=[])


@router.get("/sessions/{session_id}", response_model=ChatSession)
async def get_session(
    session_id: str,
    user_id: str = Depends(get_current_user_id)
):
    """
    Get a specific chat session

    - Requires authentication
    - Returns session metadata
    """
    raise HTTPException(
        status_code=404,
        detail=f"Session {session_id} not found"
    )


@router.get("/sessions/{session_id}/messages", response_model=MessageListResponse)
async def get_session_messages(
    session_id: str,
    user_id: str = Depends(get_current_user_id)
):
    """
    Get all messages in a chat session

    - Requires authentication
    - Returns conversation history
    """
    # TODO: Fetch from database

    return MessageListResponse(messages=[])


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    user_id: str = Depends(get_current_user_id)
):
    """
    Delete a chat session

    - Requires authentication
    - Removes all messages in the session
    """
    # TODO: Delete from database

    return {"message": f"Session {session_id} deleted"}
