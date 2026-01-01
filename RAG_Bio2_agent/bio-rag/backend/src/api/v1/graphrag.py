"""GraphRAG API Endpoints"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

from src.services.graph_rag import get_graph_rag_service, GraphRAGService
from src.services.graph_db import get_graph_service, GraphDBService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/graphrag", tags=["GraphRAG"])


# ==================== Request/Response Models ====================

class GraphRAGQueryRequest(BaseModel):
    """Request for GraphRAG query"""
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=50)
    use_graph: bool = Field(default=True)
    graph_depth: int = Field(default=2, ge=1, le=4)
    search_mode: str = Field(default='hybrid', pattern='^(hybrid|dense|sparse)$')


class GraphSourceResponse(BaseModel):
    """Source document with graph context"""
    pmid: str
    title: str
    abstract: str
    relevance_score: float
    source_type: str
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    citing_papers: List[str] = []
    cited_by: List[str] = []
    related_keywords: List[str] = []
    graph_context: Dict[str, Any] = {}


class GraphRAGQueryResponse(BaseModel):
    """Response from GraphRAG query"""
    answer: str
    sources: List[GraphSourceResponse]
    confidence: float
    processing_time_ms: float
    vector_results_count: int
    graph_results_count: int
    graph_enabled: bool


class IndexPaperRequest(BaseModel):
    """Request to index a paper to the graph"""
    pmid: str
    title: str
    abstract: Optional[str] = None
    authors: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    cited_pmids: Optional[List[str]] = None
    journal: Optional[str] = None
    publication_date: Optional[str] = None


class IndexPapersRequest(BaseModel):
    """Request to batch index papers"""
    papers: List[IndexPaperRequest]


class GraphStatsResponse(BaseModel):
    """Graph database statistics"""
    status: str
    papers: int = 0
    authors: int = 0
    keywords: int = 0
    citations: int = 0
    authorships: int = 0
    keyword_links: int = 0


class RelatedPapersRequest(BaseModel):
    """Request for related papers"""
    pmid: str
    relation_type: str = Field(
        default='all',
        pattern='^(all|citing|cited|co_cited|keyword)$'
    )
    limit: int = Field(default=10, ge=1, le=50)


class RelatedPapersResponse(BaseModel):
    """Related papers response"""
    pmid: str
    relation_type: str
    papers: List[Dict[str, Any]]


class CitationPathRequest(BaseModel):
    """Request for citation path between papers"""
    from_pmid: str
    to_pmid: str
    max_depth: int = Field(default=4, ge=1, le=6)


class CitationPathResponse(BaseModel):
    """Citation path response"""
    from_pmid: str
    to_pmid: str
    path: List[Dict[str, Any]]
    path_length: int


# ==================== Dependency Injection ====================

def get_graph_rag() -> GraphRAGService:
    """Get GraphRAG service instance"""
    return get_graph_rag_service()


def get_graph_db() -> GraphDBService:
    """Get Graph DB service instance"""
    return get_graph_service()


# ==================== Endpoints ====================

@router.post("/query", response_model=GraphRAGQueryResponse)
async def graphrag_query(
    request: GraphRAGQueryRequest,
    service: GraphRAGService = Depends(get_graph_rag)
):
    """
    Execute a GraphRAG query

    Combines vector search with knowledge graph traversal for enhanced retrieval.
    """
    try:
        response = await service.query(
            question=request.question,
            top_k=request.top_k,
            use_graph=request.use_graph,
            graph_depth=request.graph_depth,
            search_mode=request.search_mode
        )

        return GraphRAGQueryResponse(
            answer=response.answer,
            sources=[
                GraphSourceResponse(
                    pmid=s.pmid,
                    title=s.title,
                    abstract=s.abstract,
                    relevance_score=s.relevance_score,
                    source_type=s.source_type,
                    vector_score=s.vector_score,
                    graph_score=s.graph_score,
                    citing_papers=s.citing_papers,
                    cited_by=s.cited_by,
                    related_keywords=s.related_keywords,
                    graph_context=s.graph_context
                )
                for s in response.sources
            ],
            confidence=response.confidence,
            processing_time_ms=response.processing_time_ms,
            vector_results_count=response.vector_results_count,
            graph_results_count=response.graph_results_count,
            graph_enabled=response.graph_enabled
        )

    except Exception as e:
        logger.error(f"GraphRAG query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index")
async def index_paper(
    request: IndexPaperRequest,
    service: GraphRAGService = Depends(get_graph_rag)
):
    """
    Index a single paper to the knowledge graph

    Creates nodes for paper, authors, keywords and their relationships.
    """
    try:
        success = service.index_paper_to_graph(
            pmid=request.pmid,
            title=request.title,
            abstract=request.abstract,
            authors=request.authors,
            keywords=request.keywords,
            cited_pmids=request.cited_pmids,
            journal=request.journal,
            publication_date=request.publication_date
        )

        if success:
            return {"status": "success", "pmid": request.pmid}
        else:
            return {"status": "partial", "pmid": request.pmid, "message": "Some operations may have failed"}

    except Exception as e:
        logger.error(f"Index paper error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index/batch")
async def index_papers_batch(
    request: IndexPapersRequest,
    service: GraphRAGService = Depends(get_graph_rag)
):
    """
    Batch index multiple papers to the knowledge graph
    """
    try:
        results = []
        for paper in request.papers:
            success = service.index_paper_to_graph(
                pmid=paper.pmid,
                title=paper.title,
                abstract=paper.abstract,
                authors=paper.authors,
                keywords=paper.keywords,
                cited_pmids=paper.cited_pmids,
                journal=paper.journal,
                publication_date=paper.publication_date
            )
            results.append({"pmid": paper.pmid, "success": success})

        success_count = sum(1 for r in results if r['success'])
        return {
            "status": "completed",
            "total": len(request.papers),
            "success": success_count,
            "failed": len(request.papers) - success_count,
            "results": results
        }

    except Exception as e:
        logger.error(f"Batch index error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=GraphStatsResponse)
async def get_graph_stats(
    service: GraphDBService = Depends(get_graph_db)
):
    """
    Get knowledge graph statistics
    """
    try:
        stats = service.get_stats()
        return GraphStatsResponse(**stats)

    except Exception as e:
        logger.error(f"Stats error: {e}")
        return GraphStatsResponse(status="error")


@router.post("/related", response_model=RelatedPapersResponse)
async def get_related_papers(
    request: RelatedPapersRequest,
    service: GraphDBService = Depends(get_graph_db)
):
    """
    Find papers related to a given paper through the knowledge graph
    """
    try:
        papers = []

        if request.relation_type in ['all', 'citing']:
            citing = service.find_citing_papers(request.pmid, request.limit)
            for p in citing:
                p['relation'] = 'citing'
            papers.extend(citing)

        if request.relation_type in ['all', 'cited']:
            cited = service.find_cited_papers(request.pmid, request.limit)
            for p in cited:
                p['relation'] = 'cited'
            papers.extend(cited)

        if request.relation_type in ['all', 'co_cited']:
            co_cited = service.find_co_cited_papers(request.pmid, request.limit)
            for p in co_cited:
                p['relation'] = 'co_cited'
            papers.extend(co_cited)

        if request.relation_type in ['all', 'keyword']:
            keyword_related = service.find_related_by_keywords(request.pmid, request.limit)
            for p in keyword_related:
                p['relation'] = 'keyword'
            papers.extend(keyword_related)

        return RelatedPapersResponse(
            pmid=request.pmid,
            relation_type=request.relation_type,
            papers=papers[:request.limit]
        )

    except Exception as e:
        logger.error(f"Related papers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/path", response_model=CitationPathResponse)
async def find_citation_path(
    request: CitationPathRequest,
    service: GraphDBService = Depends(get_graph_db)
):
    """
    Find the shortest citation path between two papers
    """
    try:
        path = service.find_citation_path(
            from_pmid=request.from_pmid,
            to_pmid=request.to_pmid,
            max_depth=request.max_depth
        )

        return CitationPathResponse(
            from_pmid=request.from_pmid,
            to_pmid=request.to_pmid,
            path=path,
            path_length=len(path) - 1 if path else 0
        )

    except Exception as e:
        logger.error(f"Citation path error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/setup")
async def setup_graph_schema(
    service: GraphDBService = Depends(get_graph_db)
):
    """
    Setup Neo4j schema (indexes and constraints)

    Should be called once when setting up the database.
    """
    try:
        service.setup_schema()
        return {"status": "success", "message": "Schema setup complete"}

    except Exception as e:
        logger.error(f"Schema setup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class VisualizationRequest(BaseModel):
    """Request for graph visualization data"""
    query: Optional[str] = None
    limit: int = Field(default=50, ge=1, le=200)


class GraphNode(BaseModel):
    """Node in the graph visualization"""
    id: str
    type: str  # 'paper', 'author', 'keyword'
    label: str
    pmid: Optional[str] = None
    title: Optional[str] = None
    name: Optional[str] = None
    term: Optional[str] = None
    journal: Optional[str] = None


class GraphEdge(BaseModel):
    """Edge in the graph visualization"""
    source: str
    target: str
    type: str  # 'cites', 'authored', 'has_keyword'
    label: str


class VisualizationResponse(BaseModel):
    """Graph visualization data response"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    status: str
    node_count: int = 0
    edge_count: int = 0
    error: Optional[str] = None


@router.post("/visualization", response_model=VisualizationResponse)
async def get_graph_visualization(
    request: VisualizationRequest,
    service: GraphDBService = Depends(get_graph_db)
):
    """
    Get graph visualization data (nodes and edges)

    Returns data formatted for 3D knowledge graph visualization.
    Includes papers, authors, and keywords with their relationships.
    """
    try:
        data = service.get_visualization_data(
            query=request.query,
            limit=request.limit
        )
        return VisualizationResponse(**data)

    except Exception as e:
        logger.error(f"Visualization error: {e}")
        return VisualizationResponse(
            nodes=[],
            edges=[],
            status="error",
            error=str(e)
        )
