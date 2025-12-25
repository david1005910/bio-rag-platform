"""Search API Endpoints"""

import time
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.data import sample_papers

router = APIRouter()


# ============== Schemas ==============

class SearchFilters(BaseModel):
    """Search filters"""
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    journals: Optional[List[str]] = None
    authors: Optional[List[str]] = None


class SearchRequest(BaseModel):
    """Search request"""
    query: str
    filters: Optional[SearchFilters] = None
    limit: int = 10
    offset: int = 0


class PaperResult(BaseModel):
    """Paper search result"""
    pmid: str
    title: str
    abstract: str
    relevance_score: float
    authors: List[str] = []
    journal: str = ""
    publication_date: Optional[str] = None
    keywords: List[str] = []


class SearchResponse(BaseModel):
    """Search response"""
    total: int
    took_ms: int
    results: List[PaperResult]


class PaperDetailResponse(BaseModel):
    """Paper detail response"""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    publication_date: Optional[str] = None
    doi: Optional[str] = None
    keywords: List[str] = []
    mesh_terms: List[str] = []


class SimilarPaperResponse(BaseModel):
    """Similar paper response"""
    pmid: str
    title: str
    similarity_score: float
    common_keywords: List[str] = []


# ============== Endpoints ==============

@router.post("/search", response_model=SearchResponse)
async def search_papers_endpoint(request: SearchRequest):
    """
    Semantic search for papers

    - Uses vector similarity for semantic matching
    - Supports filters for year, journal, authors
    - Returns top-K most relevant papers
    """
    start_time = time.time()

    # Convert filters to dict if present
    filters = None
    if request.filters:
        filters = request.filters.model_dump()

    # Search sample papers
    total, results = sample_papers.search_papers(
        query=request.query,
        limit=request.limit,
        offset=request.offset,
        filters=filters
    )

    took_ms = int((time.time() - start_time) * 1000)

    return SearchResponse(
        total=total,
        took_ms=took_ms,
        results=[PaperResult(**r) for r in results]
    )


@router.get("/papers/{pmid}", response_model=PaperDetailResponse)
async def get_paper(pmid: str):
    """
    Get paper details by PMID

    - Returns full paper metadata
    - Includes abstract, authors, keywords
    """
    paper = sample_papers.get_paper_by_pmid(pmid)

    if not paper:
        raise HTTPException(
            status_code=404,
            detail=f"Paper with PMID {pmid} not found"
        )

    return PaperDetailResponse(**paper)


@router.get("/papers/{pmid}/similar", response_model=List[SimilarPaperResponse])
async def get_similar_papers_endpoint(
    pmid: str,
    limit: int = Query(default=5, ge=1, le=20)
):
    """
    Get similar papers

    - Uses cosine similarity on embeddings
    - Returns top-K most similar papers
    """
    similar = sample_papers.get_similar_papers(pmid, limit)
    return [SimilarPaperResponse(**s) for s in similar]


@router.get("/papers/{pmid}/ask")
async def ask_about_paper(
    pmid: str,
    question: str = Query(..., min_length=5)
):
    """
    Ask a question about a specific paper

    - Uses RAG with the paper as context
    - Returns AI-generated answer with citations
    """
    # TODO: Implement with RAGService

    return {
        "answer": "This feature is coming soon.",
        "sources": [pmid]
    }
