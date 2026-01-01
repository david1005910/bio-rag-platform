"""GraphRAG Service - Combining Vector Search with Graph Traversal"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from src.services.graph_db import GraphDBService, get_graph_service, GraphSearchResult
from src.services.storage.vector_store import VectorStore
from src.services.ai_chat import AIService, ChatSource, get_ai_service

logger = logging.getLogger(__name__)


@dataclass
class GraphRAGSource:
    """Enhanced source with graph context"""
    pmid: str
    title: str
    abstract: str
    relevance_score: float
    source_type: str  # 'vector', 'graph', 'hybrid'
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    graph_context: Dict[str, Any] = field(default_factory=dict)
    # Graph relationships
    citing_papers: List[str] = field(default_factory=list)
    cited_by: List[str] = field(default_factory=list)
    related_keywords: List[str] = field(default_factory=list)


@dataclass
class GraphRAGResponse:
    """Response from GraphRAG query"""
    answer: str
    sources: List[GraphRAGSource]
    confidence: float
    processing_time_ms: float
    vector_results_count: int
    graph_results_count: int
    graph_enabled: bool


class GraphRAGService:
    """
    GraphRAG Service - Enhanced RAG with Knowledge Graph

    Combines:
    1. Dense vector search (Qdrant) for semantic similarity
    2. Graph traversal (Neo4j) for structural relationships
    3. Reranking based on combined scores
    """

    def __init__(
        self,
        graph_service: Optional[GraphDBService] = None,
        vector_service: Optional[VectorStore] = None,
        ai_service: Optional[AIService] = None,
        graph_weight: float = 0.3,  # Weight for graph score (0-1)
        vector_weight: float = 0.7,  # Weight for vector score (0-1)
    ):
        self.graph_service = graph_service or get_graph_service()
        self.vector_service = vector_service
        self.ai_service = ai_service or get_ai_service()
        self.graph_weight = graph_weight
        self.vector_weight = vector_weight

    async def query(
        self,
        question: str,
        top_k: int = 10,
        use_graph: bool = True,
        graph_depth: int = 2,
        search_mode: str = 'hybrid'
    ) -> GraphRAGResponse:
        """
        Execute GraphRAG query

        Args:
            question: User's question
            top_k: Number of results to return
            use_graph: Whether to use graph enhancement
            graph_depth: How deep to traverse the graph
            search_mode: 'hybrid', 'dense', or 'sparse'

        Returns:
            GraphRAGResponse with answer and enriched sources
        """
        import time
        start_time = time.time()

        # Step 1: Vector Search
        vector_results = await self._vector_search(question, top_k * 2, search_mode)
        logger.info(f"Vector search returned {len(vector_results)} results")

        # Step 2: Graph Enhancement (if enabled and graph is available)
        graph_results = []
        if use_graph:
            seed_pmids = [r['pmid'] for r in vector_results[:5]]  # Top 5 as seeds
            graph_results = self._graph_search(seed_pmids, graph_depth, top_k)
            logger.info(f"Graph search returned {len(graph_results)} results")

        # Step 3: Merge and Rerank
        merged_sources = self._merge_results(vector_results, graph_results, top_k)

        # Step 4: Enrich with graph context
        enriched_sources = self._enrich_with_graph_context(merged_sources)

        # Step 5: Generate answer with AI
        chat_sources = [
            ChatSource(
                pmid=s.pmid,
                title=s.title,
                abstract=s.abstract,
                relevance=s.relevance_score
            )
            for s in enriched_sources
        ]

        chat_response = await self.ai_service.chat_with_context(
            question=question,
            sources=chat_sources
        )

        processing_time = (time.time() - start_time) * 1000

        return GraphRAGResponse(
            answer=chat_response.answer,
            sources=enriched_sources,
            confidence=chat_response.confidence,
            processing_time_ms=processing_time,
            vector_results_count=len(vector_results),
            graph_results_count=len(graph_results),
            graph_enabled=use_graph and len(graph_results) > 0
        )

    async def _vector_search(
        self,
        query: str,
        top_k: int,
        search_mode: str
    ) -> List[Dict[str, Any]]:
        """Perform vector database search using HybridVectorStore"""
        if not self.vector_service:
            # Import here to avoid circular imports
            try:
                from src.api.v1.vectordb import get_vector_store
                self.vector_service = get_vector_store()
                logger.info(f"VectorStore initialized with {self.vector_service.get_stats().get('vectors_count', 0)} vectors")
            except Exception as e:
                logger.warning(f"Could not initialize VectorStore: {e}")
                return []

        try:
            # Use HybridVectorStore async search
            raw_results = await self.vector_service.search(
                query=query,
                top_k=top_k,
                mode=search_mode,
                dense_weight=0.7
            )

            # Transform results to expected format (pmid at top level)
            results = []
            for r in raw_results or []:
                metadata = r.get('metadata', {})
                results.append({
                    'pmid': metadata.get('pmid', ''),
                    'title': metadata.get('title', ''),
                    'text': r.get('text', ''),
                    'abstract': r.get('text', ''),
                    'score': r.get('score', 0),
                    'dense_score': r.get('dense_score'),
                    'sparse_score': r.get('sparse_score'),
                })
            return results
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    def _graph_search(
        self,
        seed_pmids: List[str],
        depth: int,
        limit: int
    ) -> List[GraphSearchResult]:
        """Perform graph-based search"""
        try:
            return self.graph_service.graph_enhanced_search(
                seed_pmids=seed_pmids,
                max_depth=depth,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Graph search error: {e}")
            return []

    def _merge_results(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[GraphSearchResult],
        top_k: int
    ) -> List[GraphRAGSource]:
        """
        Merge vector and graph results with weighted scoring

        Uses Reciprocal Rank Fusion (RRF) for combining rankings
        """
        scores: Dict[str, Dict[str, Any]] = {}
        k = 60  # RRF constant

        # Process vector results
        for rank, result in enumerate(vector_results):
            pmid = result.get('pmid', '')
            if not pmid:
                continue

            vector_score = result.get('score', 0.5)
            rrf_score = 1 / (k + rank + 1)

            scores[pmid] = {
                'pmid': pmid,
                'title': result.get('title', ''),
                'abstract': result.get('text', result.get('abstract', '')),
                'vector_score': vector_score,
                'vector_rrf': rrf_score * self.vector_weight,
                'graph_score': 0,
                'graph_rrf': 0,
                'source_type': 'vector'
            }

        # Process graph results
        for rank, result in enumerate(graph_results):
            pmid = result.pmid
            graph_score = result.relevance_score
            rrf_score = 1 / (k + rank + 1)

            if pmid in scores:
                scores[pmid]['graph_score'] = graph_score
                scores[pmid]['graph_rrf'] = rrf_score * self.graph_weight
                scores[pmid]['source_type'] = 'hybrid'
            else:
                scores[pmid] = {
                    'pmid': pmid,
                    'title': result.title,
                    'abstract': '',  # May need to fetch
                    'vector_score': 0,
                    'vector_rrf': 0,
                    'graph_score': graph_score,
                    'graph_rrf': rrf_score * self.graph_weight,
                    'source_type': 'graph'
                }

        # Calculate combined scores and sort
        for pmid, data in scores.items():
            data['combined_score'] = data['vector_rrf'] + data['graph_rrf']

        sorted_results = sorted(
            scores.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )[:top_k]

        # Convert to GraphRAGSource
        return [
            GraphRAGSource(
                pmid=r['pmid'],
                title=r['title'],
                abstract=r['abstract'],
                relevance_score=r['combined_score'],
                source_type=r['source_type'],
                vector_score=r['vector_score'],
                graph_score=r['graph_score']
            )
            for r in sorted_results
        ]

    def _enrich_with_graph_context(
        self,
        sources: List[GraphRAGSource]
    ) -> List[GraphRAGSource]:
        """Enrich sources with additional graph context"""
        for source in sources:
            try:
                # Get citing papers
                citing = self.graph_service.find_citing_papers(source.pmid, limit=3)
                source.citing_papers = [p['pmid'] for p in citing if p.get('pmid')]

                # Get cited papers
                cited = self.graph_service.find_cited_papers(source.pmid, limit=3)
                source.cited_by = [p['pmid'] for p in cited if p.get('pmid')]

                # Get related keywords
                related = self.graph_service.find_related_by_keywords(source.pmid, limit=5)
                if related:
                    all_keywords = []
                    for r in related:
                        kws = r.get('shared_keywords', [])
                        if kws:
                            all_keywords.extend(kws)
                    source.related_keywords = list(set(all_keywords))[:5]

                source.graph_context = {
                    'citation_count': len(source.citing_papers) + len(source.cited_by),
                    'keyword_connections': len(source.related_keywords)
                }

            except Exception as e:
                logger.debug(f"Could not enrich {source.pmid}: {e}")

        return sources

    def index_paper_to_graph(
        self,
        pmid: str,
        title: str,
        abstract: str = None,
        authors: List[str] = None,
        keywords: List[str] = None,
        cited_pmids: List[str] = None,
        journal: str = None,
        publication_date: str = None
    ) -> bool:
        """
        Index a paper and its relationships to the graph database

        Args:
            pmid: PubMed ID
            title: Paper title
            abstract: Paper abstract
            authors: List of author names
            keywords: List of keywords/MeSH terms
            cited_pmids: List of PMIDs this paper cites
            journal: Journal name
            publication_date: Publication date

        Returns:
            True if successful
        """
        from src.services.graph_db import PaperNode, CitationRelation

        try:
            # Create paper node
            paper = PaperNode(
                pmid=pmid,
                title=title,
                abstract=abstract,
                journal=journal,
                publication_date=publication_date
            )
            self.graph_service.create_paper(paper)

            # Link authors
            if authors:
                for i, author in enumerate(authors):
                    self.graph_service.link_author_to_paper(author, pmid, position=i)

            # Link keywords
            if keywords:
                for keyword in keywords:
                    self.graph_service.link_keyword_to_paper(keyword, pmid)

            # Create citation relationships
            if cited_pmids:
                citations = [
                    CitationRelation(citing_pmid=pmid, cited_pmid=cited)
                    for cited in cited_pmids
                ]
                self.graph_service.create_citations_batch(citations)

            logger.info(f"Indexed paper {pmid} to graph with {len(authors or [])} authors, "
                       f"{len(keywords or [])} keywords, {len(cited_pmids or [])} citations")
            return True

        except Exception as e:
            logger.error(f"Failed to index paper {pmid} to graph: {e}")
            return False

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph database statistics"""
        return self.graph_service.get_stats()


# Global service instance
_graph_rag_service: Optional[GraphRAGService] = None


def get_graph_rag_service() -> GraphRAGService:
    """Get or create GraphRAG service instance"""
    global _graph_rag_service
    if _graph_rag_service is None:
        _graph_rag_service = GraphRAGService()
    return _graph_rag_service
