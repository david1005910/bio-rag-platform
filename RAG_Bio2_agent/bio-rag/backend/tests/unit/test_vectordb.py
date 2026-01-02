"""Tests for VectorDB API - Hybrid Search with Dense (Qdrant) + Sparse (SPLADE)"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.api.v1.vectordb import (
    PaperForVectorDB,
    SavePapersRequest,
    SavePapersResponse,
    VectorDBStatsResponse,
    SearchVectorDBRequest,
    VectorSearchResult,
    SearchVectorDBResponse,
    SPLADESearch,
    QdrantDenseSearch,
    HybridVectorStore,
    chunk_text,
    get_vector_store,
    router
)


# ============== Schema Tests ==============

class TestSchemas:
    """Test Pydantic schemas"""

    def test_paper_for_vectordb(self):
        """Test PaperForVectorDB schema"""
        paper = PaperForVectorDB(
            pmid="12345678",
            title="CRISPR Gene Editing",
            abstract="This study explores CRISPR-based gene editing.",
            authors=["Kim J", "Lee S"],
            journal="Nature",
            publication_date="2024-01-15",
            keywords=["CRISPR", "gene editing"]
        )
        assert paper.pmid == "12345678"
        assert paper.title == "CRISPR Gene Editing"
        assert len(paper.authors) == 2
        assert len(paper.keywords) == 2

    def test_paper_for_vectordb_defaults(self):
        """Test PaperForVectorDB with default values"""
        paper = PaperForVectorDB(
            pmid="12345678",
            title="Test Paper",
            abstract="Abstract"
        )
        assert paper.authors == []
        assert paper.journal == ""
        assert paper.publication_date is None
        assert paper.keywords == []

    def test_save_papers_request(self):
        """Test SavePapersRequest schema"""
        request = SavePapersRequest(
            papers=[
                PaperForVectorDB(pmid="1", title="Paper 1", abstract="Abstract 1"),
                PaperForVectorDB(pmid="2", title="Paper 2", abstract="Abstract 2")
            ]
        )
        assert len(request.papers) == 2

    def test_save_papers_response(self):
        """Test SavePapersResponse schema"""
        response = SavePapersResponse(
            saved_count=5,
            total_chunks=15,
            processing_time_ms=500,
            paper_ids=["1", "2", "3", "4", "5"]
        )
        assert response.saved_count == 5
        assert response.total_chunks == 15
        assert len(response.paper_ids) == 5

    def test_vectordb_stats_response(self):
        """Test VectorDBStatsResponse schema"""
        response = VectorDBStatsResponse(
            collection_name="biomedical_papers",
            vectors_count=100,
            status="ready",
            search_mode="hybrid",
            dense_engine="qdrant_local",
            sparse_engine="splade",
            splade_indexed=True,
            splade_vocab_size=5000,
            with_embeddings=100,
            qdrant_status="qdrant_local"
        )
        assert response.collection_name == "biomedical_papers"
        assert response.search_mode == "hybrid"
        assert response.splade_indexed is True

    def test_search_vectordb_request(self):
        """Test SearchVectorDBRequest schema"""
        request = SearchVectorDBRequest(
            query="CRISPR gene therapy",
            top_k=10,
            search_mode="hybrid",
            dense_weight=0.8
        )
        assert request.query == "CRISPR gene therapy"
        assert request.top_k == 10
        assert request.dense_weight == 0.8

    def test_search_vectordb_request_defaults(self):
        """Test SearchVectorDBRequest with defaults"""
        request = SearchVectorDBRequest(query="test query")
        assert request.top_k == 5
        assert request.search_mode == "hybrid"
        assert request.dense_weight == 0.7

    def test_vector_search_result(self):
        """Test VectorSearchResult schema"""
        result = VectorSearchResult(
            pmid="12345678",
            title="Test Paper",
            text="This is the text content...",
            score=0.95,
            dense_score=0.92,
            sparse_score=5.5,
            section="abstract"
        )
        assert result.pmid == "12345678"
        assert result.score == 0.95
        assert result.dense_score == 0.92
        assert result.sparse_score == 5.5

    def test_search_vectordb_response(self):
        """Test SearchVectorDBResponse schema"""
        response = SearchVectorDBResponse(
            results=[
                VectorSearchResult(
                    pmid="1", title="Paper 1", text="Text", score=0.9,
                    section="abstract"
                )
            ],
            took_ms=150,
            search_mode="hybrid"
        )
        assert len(response.results) == 1
        assert response.took_ms == 150


# ============== SPLADE Search Tests ==============

class TestSPLADESearch:
    """Test SPLADESearch class"""

    @pytest.fixture
    def splade(self):
        """Create SPLADESearch instance"""
        return SPLADESearch(k1=1.5, b=0.75)

    def test_init(self, splade):
        """Test SPLADE initialization"""
        assert splade.k1 == 1.5
        assert splade.b == 0.75
        assert splade.n_docs == 0

    def test_tokenize_basic(self, splade):
        """Test basic tokenization"""
        tokens = splade._tokenize("CRISPR gene editing")
        assert "crispr" in tokens
        assert "gene" in tokens
        assert "editing" in tokens

    def test_tokenize_with_hyphen(self, splade):
        """Test tokenization preserves hyphens"""
        tokens = splade._tokenize("CRISPR-Cas9 system")
        assert "crispr-cas9" in tokens
        assert "system" in tokens

    def test_tokenize_removes_punctuation(self, splade):
        """Test tokenization removes punctuation"""
        tokens = splade._tokenize("Hello, world! Test.")
        assert "hello" in tokens
        assert "world" in tokens
        # Commas and periods should be removed
        assert "hello," not in tokens

    def test_tokenize_filters_short_tokens(self, splade):
        """Test tokenization filters single character tokens"""
        tokens = splade._tokenize("A B test")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "test" in tokens

    def test_fit_builds_index(self, splade):
        """Test fit builds SPLADE index"""
        documents = [
            "CRISPR gene editing for cancer therapy",
            "Immunotherapy advances in oncology treatment",
            "Machine learning in drug discovery"
        ]
        doc_ids = ["1", "2", "3"]

        splade.fit(documents, doc_ids)

        assert splade.n_docs == 3
        assert len(splade.doc_lens) == 3
        assert "crispr" in splade.idf
        assert "cancer" in splade.idf
        assert len(splade.term_weights) == 3

    def test_fit_calculates_idf(self, splade):
        """Test IDF calculation"""
        documents = [
            "cancer cancer cancer",
            "cancer therapy",
            "therapy treatment"
        ]
        doc_ids = ["1", "2", "3"]

        splade.fit(documents, doc_ids)

        # "cancer" appears in 2 docs, "therapy" appears in 2 docs
        assert "cancer" in splade.idf
        assert "therapy" in splade.idf
        # IDF for terms appearing in fewer docs should be higher
        assert splade.idf["treatment"] > splade.idf["cancer"]

    def test_score_single_term(self, splade):
        """Test BM25 scoring with single term"""
        documents = ["crispr gene editing", "cancer therapy", "machine learning"]
        doc_ids = ["1", "2", "3"]
        splade.fit(documents, doc_ids)

        query_terms = [("crispr", 2.0)]
        score, term_scores = splade.score(query_terms, 0, documents[0])

        assert score > 0
        assert "crispr" in term_scores

    def test_score_multiple_terms(self, splade):
        """Test BM25 scoring with multiple terms"""
        documents = ["crispr gene editing therapy", "cancer therapy", "machine learning"]
        doc_ids = ["1", "2", "3"]
        splade.fit(documents, doc_ids)

        query_terms = [("crispr", 2.0), ("gene", 1.5), ("therapy", 1.0)]
        score, term_scores = splade.score(query_terms, 0, documents[0])

        assert score > 0
        assert len(term_scores) >= 2  # At least crispr and gene should match

    def test_score_phrase_match(self, splade):
        """Test phrase match gets boosted score"""
        documents = ["CRISPR gene editing therapy", "gene therapy", "cancer treatment"]
        doc_ids = ["1", "2", "3"]
        splade.fit(documents, doc_ids)

        query_terms = [("gene editing", 2.0)]  # Multi-word term
        score, term_scores = splade.score(query_terms, 0, documents[0])

        assert score > 0
        assert "gene editing" in term_scores

    def test_score_no_match(self, splade):
        """Test score when no terms match"""
        documents = ["crispr gene editing", "cancer therapy", "machine learning"]
        doc_ids = ["1", "2", "3"]
        splade.fit(documents, doc_ids)

        query_terms = [("immunotherapy", 2.0)]
        score, term_scores = splade.score(query_terms, 0, documents[0])

        assert score == 0
        assert len(term_scores) == 0

    @pytest.mark.asyncio
    async def test_expand_query_without_api_key(self, splade):
        """Test query expansion without API key"""
        with patch('src.api.v1.vectordb.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = ""
            terms = await splade._expand_query("CRISPR gene editing")

        # Should return original tokens with weight 2.0
        assert len(terms) >= 3
        assert any(t[0] == "crispr" for t in terms)
        assert all(t[1] == 2.0 for t in terms)  # All original terms have weight 2.0

    @pytest.mark.asyncio
    async def test_search(self, splade):
        """Test SPLADE search"""
        documents = [
            "CRISPR gene editing for cancer therapy",
            "Immunotherapy advances in oncology treatment",
            "Machine learning in drug discovery"
        ]
        doc_ids = ["1", "2", "3"]
        splade.fit(documents, doc_ids)

        with patch.object(splade, '_expand_query') as mock_expand:
            mock_expand.return_value = [("crispr", 2.0), ("gene", 1.5), ("editing", 1.0)]
            results = await splade.search("CRISPR gene editing", documents, doc_ids, top_k=2)

        assert len(results) >= 1
        assert results[0]["doc_id"] == "1"  # First doc should match best
        assert results[0]["score"] > 0


# ============== Qdrant Dense Search Tests ==============

class TestQdrantDenseSearch:
    """Test QdrantDenseSearch class"""

    def test_init_fallback_to_none(self):
        """Test initialization falls back when Qdrant unavailable"""
        with patch('qdrant_client.QdrantClient') as MockClient:
            MockClient.side_effect = Exception("Connection failed")

            with patch('src.api.v1.vectordb.settings') as mock_settings:
                mock_settings.QDRANT_HOST = "localhost"
                mock_settings.QDRANT_PORT = 6333
                mock_settings.QDRANT_COLLECTION = "test_collection"

                search = QdrantDenseSearch()

        # Should fall back to no Qdrant mode or local mode
        assert search.qdrant_mode in ["none", "local", "server"]

    def test_get_collection_info_no_qdrant(self):
        """Test get_collection_info when Qdrant is not available"""
        with patch.object(QdrantDenseSearch, '_init_qdrant') as mock_init:
            search = QdrantDenseSearch.__new__(QdrantDenseSearch)
            search.qdrant_client = None
            search.collection_name = "test_collection"
            search.embedding_dim = 1536
            search.use_qdrant = False
            search.qdrant_mode = "none"

        info = search.get_collection_info()
        assert info["status"] == "in_memory"
        assert info["mode"] == "none"

    def test_get_collection_info_with_qdrant(self):
        """Test get_collection_info when Qdrant is available"""
        with patch.object(QdrantDenseSearch, '_init_qdrant'):
            search = QdrantDenseSearch.__new__(QdrantDenseSearch)
            search.collection_name = "test_collection"
            search.embedding_dim = 1536
            search.use_qdrant = True
            search.qdrant_mode = "local"

            # Mock qdrant_client
            mock_client = MagicMock()
            mock_info = MagicMock()
            mock_info.vectors_count = 100
            mock_info.points_count = 50
            mock_client.get_collection.return_value = mock_info
            search.qdrant_client = mock_client

        info = search.get_collection_info()
        assert info["status"] == "qdrant_local"
        assert info["vectors_count"] == 100
        assert info["mode"] == "local"


# ============== Hybrid Vector Store Tests ==============

class TestHybridVectorStore:
    """Test HybridVectorStore class"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create HybridVectorStore with mocked components"""
        with patch.object(HybridVectorStore, '_sync_from_qdrant'):
            with patch.object(QdrantDenseSearch, '_init_qdrant'):
                store = HybridVectorStore()
                store.qdrant_dense.use_qdrant = False
                store.qdrant_dense.qdrant_mode = "none"
                return store

    def test_init(self, mock_vector_store):
        """Test HybridVectorStore initialization"""
        assert mock_vector_store.documents == []
        assert mock_vector_store.embedding_dim == 1536
        assert mock_vector_store._splade_fitted is False

    def test_rebuild_sparse_index(self, mock_vector_store):
        """Test rebuilding SPLADE index"""
        mock_vector_store.documents = [
            {"id": "1", "text": "CRISPR gene editing", "embedding": None, "metadata": {}},
            {"id": "2", "text": "Cancer therapy", "embedding": None, "metadata": {}}
        ]

        mock_vector_store._rebuild_sparse_index()

        assert mock_vector_store._splade_fitted is True
        assert mock_vector_store.splade.n_docs == 2

    @pytest.mark.asyncio
    async def test_get_embedding_no_api_key(self, mock_vector_store):
        """Test get_embedding without API key"""
        with patch('src.api.v1.vectordb.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = ""
            result = await mock_vector_store.get_embedding("test text")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_embedding_success(self, mock_vector_store):
        """Test successful embedding generation"""
        mock_embedding = [0.1] * 1536

        with patch('src.api.v1.vectordb.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "test_key"

            with patch('aiohttp.ClientSession') as MockSession:
                mock_response = MagicMock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={
                    "data": [{"embedding": mock_embedding}]
                })

                mock_context = MagicMock()
                mock_context.__aenter__ = AsyncMock(return_value=mock_response)
                mock_context.__aexit__ = AsyncMock()

                mock_session_instance = MagicMock()
                mock_session_instance.post.return_value = mock_context
                mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
                mock_session_instance.__aexit__ = AsyncMock()

                MockSession.return_value = mock_session_instance

                result = await mock_vector_store.get_embedding("test text")

        assert result is not None
        assert len(result) == 1536

    @pytest.mark.asyncio
    async def test_get_batch_embeddings_no_api_key(self, mock_vector_store):
        """Test batch embeddings without API key"""
        with patch('src.api.v1.vectordb.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = ""
            results = await mock_vector_store.get_batch_embeddings(["text1", "text2"])

        assert results == [None, None]

    def test_normalize_scores(self, mock_vector_store):
        """Test score normalization"""
        scores = [1.0, 5.0, 10.0]
        normalized = mock_vector_store._normalize_scores(scores)

        assert normalized[0] == 0.0  # Min
        assert normalized[2] == 1.0  # Max
        assert 0 < normalized[1] < 1  # Middle

    def test_normalize_scores_empty(self, mock_vector_store):
        """Test normalization with empty list"""
        normalized = mock_vector_store._normalize_scores([])
        assert normalized == []

    def test_normalize_scores_same_values(self, mock_vector_store):
        """Test normalization when all values are same"""
        normalized = mock_vector_store._normalize_scores([5.0, 5.0, 5.0])
        assert all(n == 1.0 for n in normalized)

    def test_normalize_dense_score(self, mock_vector_store):
        """Test dense score normalization"""
        # Positive score
        assert mock_vector_store._normalize_dense_score(0.8) == 0.8
        assert mock_vector_store._normalize_dense_score(1.0) == 1.0

        # Negative score (cosine similarity can be negative)
        result = mock_vector_store._normalize_dense_score(-0.5)
        assert 0 <= result <= 1

    def test_normalize_sparse_score(self, mock_vector_store):
        """Test sparse score normalization"""
        assert mock_vector_store._normalize_sparse_score(0) == 0.0
        assert mock_vector_store._normalize_sparse_score(-1) == 0.0

        # Positive scores get scaled
        result = mock_vector_store._normalize_sparse_score(5.0)
        assert result > 0
        assert result <= 30.0

    @pytest.mark.asyncio
    async def test_search_dense_empty(self, mock_vector_store):
        """Test dense search with no documents"""
        results = await mock_vector_store.search_dense("test query", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_sparse_empty(self, mock_vector_store):
        """Test sparse search with no documents"""
        results = await mock_vector_store.search_sparse("test query", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_sparse_not_fitted(self, mock_vector_store):
        """Test sparse search when SPLADE not fitted"""
        mock_vector_store.documents = [
            {"id": "1", "text": "Test", "embedding": None, "metadata": {}}
        ]
        # SPLADE not fitted

        results = await mock_vector_store.search_sparse("test query", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_hybrid_empty(self, mock_vector_store):
        """Test hybrid search with no documents"""
        results = await mock_vector_store.search_hybrid("test query", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_modes(self, mock_vector_store):
        """Test different search modes"""
        with patch.object(mock_vector_store, 'search_dense', return_value=[]) as mock_dense:
            await mock_vector_store.search("query", mode="dense")
            mock_dense.assert_called_once()

        with patch.object(mock_vector_store, 'search_sparse', return_value=[]) as mock_sparse:
            await mock_vector_store.search("query", mode="sparse")
            mock_sparse.assert_called_once()

        with patch.object(mock_vector_store, 'search_hybrid', return_value=[]) as mock_hybrid:
            await mock_vector_store.search("query", mode="hybrid")
            mock_hybrid.assert_called_once()

    def test_get_stats(self, mock_vector_store):
        """Test get_stats"""
        mock_vector_store.documents = [
            {"id": "1", "text": "Test", "embedding": np.array([0.1] * 1536), "metadata": {}},
            {"id": "2", "text": "Test2", "embedding": None, "metadata": {}}
        ]
        mock_vector_store._splade_fitted = True
        mock_vector_store.splade.idf = {"term1": 1.0, "term2": 0.5}

        stats = mock_vector_store.get_stats()

        assert stats["vectors_count"] == 2
        assert stats["with_embeddings"] == 1
        assert stats["splade_indexed"] is True
        assert stats["splade_vocab_size"] == 2
        assert stats["status"] == "ready"

    def test_get_papers(self, mock_vector_store):
        """Test get_papers"""
        mock_vector_store.documents = [
            {
                "id": "doc1",
                "text": "Abstract text here",
                "embedding": None,
                "metadata": {
                    "pmid": "12345678",
                    "title": "Test Paper",
                    "journal": "Nature",
                    "authors": ["Kim J", "Lee S"],
                    "keywords": ["CRISPR", "gene editing"]
                }
            },
            {
                "id": "doc2",
                "text": "Another abstract",
                "embedding": None,
                "metadata": {
                    "pmid": "87654321",
                    "title": "Another Paper",
                    "journal": "Science",
                    "authors": "Park J, Choi M",  # String format
                    "keywords": "cancer, therapy"  # String format
                }
            }
        ]

        papers = mock_vector_store.get_papers()

        assert len(papers) == 2
        # Check first paper
        paper1 = next(p for p in papers if p["pmid"] == "12345678")
        assert paper1["title"] == "Test Paper"
        assert paper1["authors"] == ["Kim J", "Lee S"]

        # Check second paper (string authors/keywords parsed)
        paper2 = next(p for p in papers if p["pmid"] == "87654321")
        assert len(paper2["authors"]) == 2
        assert len(paper2["keywords"]) == 2

    def test_get_papers_deduplicates_by_pmid(self, mock_vector_store):
        """Test get_papers deduplicates by PMID"""
        mock_vector_store.documents = [
            {"id": "1", "text": "Chunk 1", "embedding": None, "metadata": {"pmid": "12345", "title": "Paper"}},
            {"id": "2", "text": "Chunk 2", "embedding": None, "metadata": {"pmid": "12345", "title": "Paper"}},
            {"id": "3", "text": "Chunk 3", "embedding": None, "metadata": {"pmid": "12345", "title": "Paper"}}
        ]

        papers = mock_vector_store.get_papers()

        assert len(papers) == 1
        assert papers[0]["pmid"] == "12345"

    def test_clear(self, mock_vector_store):
        """Test clear"""
        mock_vector_store.documents = [{"id": "1", "text": "Test", "embedding": None, "metadata": {}}]
        mock_vector_store._splade_fitted = True

        mock_vector_store.clear()

        assert mock_vector_store.documents == []
        assert mock_vector_store._splade_fitted is False


# ============== Utility Function Tests ==============

class TestChunkText:
    """Test chunk_text utility function"""

    def test_empty_text(self):
        """Test chunking empty text"""
        chunks = chunk_text("")
        assert chunks == []

    def test_short_text(self):
        """Test text shorter than chunk size"""
        text = "This is a short text."
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text(self):
        """Test text longer than chunk size"""
        # Create text with 1000 words
        words = ["word"] * 1000
        text = " ".join(words)

        chunks = chunk_text(text, chunk_size=500, overlap=100)

        assert len(chunks) > 1
        # Check overlap exists between chunks
        chunk1_words = chunks[0].split()
        chunk2_words = chunks[1].split()
        # Last 100 words of chunk1 should appear in chunk2
        assert len(set(chunk1_words[-100:]) & set(chunk2_words[:100])) > 0

    def test_chunk_size(self):
        """Test chunks respect size limit"""
        words = ["word"] * 1500
        text = " ".join(words)

        chunks = chunk_text(text, chunk_size=500, overlap=100)

        for chunk in chunks:
            word_count = len(chunk.split())
            assert word_count <= 500


class TestGetVectorStore:
    """Test get_vector_store singleton"""

    def test_singleton(self):
        """Test get_vector_store returns singleton"""
        import src.api.v1.vectordb as vectordb_module

        # Reset singleton
        vectordb_module._vector_store = None

        with patch.object(HybridVectorStore, '_sync_from_qdrant'):
            with patch.object(QdrantDenseSearch, '_init_qdrant'):
                store1 = get_vector_store()
                store2 = get_vector_store()

        assert store1 is store2


# ============== API Endpoint Tests ==============

class TestVectorDBEndpoints:
    """Test VectorDB API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router, prefix="/vectordb")
        return TestClient(app)

    @pytest.fixture
    def mock_store(self):
        """Mock vector store"""
        with patch('src.api.v1.vectordb.get_vector_store') as mock:
            store = MagicMock()
            store.documents = []
            store._splade_fitted = False
            store.splade = MagicMock()
            store.splade.idf = {}
            store.qdrant_dense = MagicMock()
            store.qdrant_dense.use_qdrant = False
            store.qdrant_dense.qdrant_mode = "none"
            store.qdrant_dense.get_collection_info.return_value = {
                "status": "in_memory", "mode": "none"
            }
            store.get_stats.return_value = {
                "collection_name": "biomedical_papers",
                "vectors_count": 0,
                "with_embeddings": 0,
                "splade_indexed": False,
                "splade_vocab_size": 0,
                "qdrant_status": "in_memory",
                "dense_engine": "in_memory",
                "sparse_engine": "none",
                "search_mode": "hybrid (In-memory dense + SPLADE sparse)",
                "status": "ready"
            }
            mock.return_value = store
            yield store

    def test_get_stats(self, client, mock_store):
        """Test GET /vectordb/stats endpoint"""
        response = client.get("/vectordb/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["collection_name"] == "biomedical_papers"
        assert data["status"] == "ready"

    def test_search_vectordb(self, client, mock_store):
        """Test POST /vectordb/search endpoint"""
        mock_store.search = AsyncMock(return_value=[
            {
                "id": "1",
                "text": "CRISPR gene editing for cancer therapy...",
                "score": 0.95,
                "dense_score": 0.92,
                "sparse_score": 5.5,
                "metadata": {
                    "pmid": "12345678",
                    "title": "CRISPR Study",
                    "section": "abstract"
                }
            }
        ])

        response = client.post("/vectordb/search", json={
            "query": "CRISPR gene editing",
            "top_k": 5,
            "search_mode": "hybrid",
            "dense_weight": 0.7
        })

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["pmid"] == "12345678"
        assert data["search_mode"] == "hybrid"

    def test_search_vectordb_empty(self, client, mock_store):
        """Test search with no results"""
        mock_store.search = AsyncMock(return_value=[])

        response = client.post("/vectordb/search", json={
            "query": "nonexistent topic",
            "top_k": 5
        })

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 0

    def test_save_papers(self, client, mock_store):
        """Test POST /vectordb/papers/save endpoint"""
        mock_store.add_documents = AsyncMock(return_value=["id1", "id2"])

        with patch('src.api.v1.vectordb.vectordb_metadata_store') as mock_metadata:
            mock_metadata.save_papers_batch = MagicMock()

            response = client.post("/vectordb/papers/save", json={
                "papers": [
                    {
                        "pmid": "12345678",
                        "title": "CRISPR Study",
                        "abstract": "This study explores CRISPR...",
                        "authors": ["Kim J"],
                        "journal": "Nature",
                        "keywords": ["CRISPR"]
                    }
                ]
            })

        assert response.status_code == 200
        data = response.json()
        assert data["saved_count"] == 1
        assert "12345678" in data["paper_ids"]

    def test_get_papers(self, client, mock_store):
        """Test GET /vectordb/papers endpoint"""
        mock_store.get_papers.return_value = [
            {
                "id": "doc1",
                "pmid": "12345678",
                "title": "Test Paper",
                "abstract": "Abstract text",
                "journal": "Nature",
                "authors": ["Kim J"],
                "keywords": ["CRISPR"],
                "indexed_at": "2024-01-15"
            }
        ]

        with patch('src.api.v1.vectordb.vectordb_metadata_store') as mock_metadata:
            mock_metadata.get_all_papers.return_value = []

            response = client.get("/vectordb/papers")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 0

    def test_get_metadata(self, client, mock_store):
        """Test GET /vectordb/metadata endpoint"""
        with patch('src.api.v1.vectordb.vectordb_metadata_store') as mock_metadata:
            mock_metadata.get_all_papers.return_value = [
                {
                    "pmid": "12345678",
                    "title": "Test Paper",
                    "abstract": "Full abstract here",
                    "journal": "Nature",
                    "authors": ["Kim J", "Lee S"],
                    "keywords": ["CRISPR"],
                    "indexed_at": "2024-01-15"
                }
            ]

            response = client.get("/vectordb/metadata")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["papers"][0]["pmid"] == "12345678"

    def test_clear_vectordb(self, client, mock_store):
        """Test DELETE /vectordb/clear endpoint"""
        mock_store.clear = MagicMock()

        response = client.delete("/vectordb/clear")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        mock_store.clear.assert_called_once()


# ============== Integration Tests ==============

class TestHybridSearchIntegration:
    """Integration tests for hybrid search"""

    @pytest.fixture
    def store_with_docs(self):
        """Create store with sample documents"""
        with patch.object(HybridVectorStore, '_sync_from_qdrant'):
            with patch.object(QdrantDenseSearch, '_init_qdrant'):
                store = HybridVectorStore()
                store.qdrant_dense.use_qdrant = False
                store.qdrant_dense.qdrant_mode = "none"

                # Add sample documents with embeddings
                store.documents = [
                    {
                        "id": "1",
                        "text": "CRISPR gene editing for cancer therapy using Cas9 nuclease",
                        "embedding": np.random.rand(1536).astype(np.float32),
                        "metadata": {"pmid": "1", "title": "CRISPR Cancer Study", "section": "abstract"}
                    },
                    {
                        "id": "2",
                        "text": "Immunotherapy advances for cancer treatment using checkpoint inhibitors",
                        "embedding": np.random.rand(1536).astype(np.float32),
                        "metadata": {"pmid": "2", "title": "Immunotherapy Study", "section": "abstract"}
                    },
                    {
                        "id": "3",
                        "text": "Machine learning for drug discovery and molecular design",
                        "embedding": np.random.rand(1536).astype(np.float32),
                        "metadata": {"pmid": "3", "title": "ML Drug Discovery", "section": "abstract"}
                    }
                ]

                # Build SPLADE index
                store._rebuild_sparse_index()

                return store

    @pytest.mark.asyncio
    async def test_sparse_search_finds_relevant_docs(self, store_with_docs):
        """Test sparse search finds relevant documents"""
        with patch.object(store_with_docs.splade, '_expand_query') as mock_expand:
            mock_expand.return_value = [("crispr", 2.0), ("gene", 1.5), ("editing", 1.0)]
            results = await store_with_docs.search_sparse("CRISPR gene editing", top_k=3)

        assert len(results) >= 1
        # First result should be the CRISPR paper
        assert results[0]["metadata"]["pmid"] == "1"

    @pytest.mark.asyncio
    async def test_dense_search_in_memory(self, store_with_docs):
        """Test in-memory dense search"""
        # Mock embedding for query
        query_embedding = np.random.rand(1536).astype(np.float32)

        with patch.object(store_with_docs, 'get_embedding', return_value=query_embedding):
            results = await store_with_docs.search_dense("cancer treatment", top_k=3)

        assert len(results) == 3  # Should return all docs
        assert all(r["search_engine"] == "in_memory" for r in results)

    @pytest.mark.asyncio
    async def test_hybrid_combines_scores(self, store_with_docs):
        """Test hybrid search combines dense and sparse scores"""
        query_embedding = np.random.rand(1536).astype(np.float32)

        with patch.object(store_with_docs, 'get_embedding', return_value=query_embedding):
            with patch.object(store_with_docs.splade, '_expand_query') as mock_expand:
                mock_expand.return_value = [("cancer", 2.0), ("therapy", 1.5)]
                results = await store_with_docs.search_hybrid("cancer therapy", top_k=3)

        assert len(results) >= 1
        # Results should have both scores
        for r in results:
            assert "dense_score" in r
            assert "sparse_score" in r
            assert "score" in r  # Hybrid score


class TestSearchScoreNormalization:
    """Test score normalization in search"""

    @pytest.fixture
    def store(self):
        """Create mock store"""
        with patch.object(HybridVectorStore, '_sync_from_qdrant'):
            with patch.object(QdrantDenseSearch, '_init_qdrant'):
                store = HybridVectorStore()
                store.qdrant_dense.use_qdrant = False
                return store

    def test_dense_score_in_range(self, store):
        """Test dense scores are normalized to [0, 1]"""
        # Test various input scores
        assert 0 <= store._normalize_dense_score(0.0) <= 1
        assert 0 <= store._normalize_dense_score(0.5) <= 1
        assert 0 <= store._normalize_dense_score(1.0) <= 1
        assert 0 <= store._normalize_dense_score(-0.5) <= 1

    def test_sparse_score_in_range(self, store):
        """Test sparse scores are normalized properly"""
        assert store._normalize_sparse_score(0) == 0.0
        assert store._normalize_sparse_score(5.0) > 0
        assert store._normalize_sparse_score(10.0) <= 30.0
