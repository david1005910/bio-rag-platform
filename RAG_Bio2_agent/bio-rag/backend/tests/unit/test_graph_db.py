"""Tests for Graph Database Service"""

import pytest
from unittest.mock import MagicMock, patch

from src.services.graph_db import (
    GraphDBService,
    PaperNode,
    AuthorNode,
    KeywordNode,
    CitationRelation,
    GraphSearchResult,
    get_graph_service
)


# ==================== Dataclass Tests ====================

class TestPaperNode:
    """Test PaperNode dataclass"""

    def test_create_paper_node(self):
        """Test creating PaperNode instance"""
        paper = PaperNode(
            pmid="12345678",
            title="CRISPR Gene Therapy",
            abstract="This study explores...",
            journal="Nature",
            publication_date="2024-01-15"
        )
        assert paper.pmid == "12345678"
        assert paper.title == "CRISPR Gene Therapy"
        assert paper.journal == "Nature"

    def test_create_paper_node_minimal(self):
        """Test creating PaperNode with minimal fields"""
        paper = PaperNode(pmid="11111111", title="Minimal Paper")
        assert paper.pmid == "11111111"
        assert paper.abstract is None
        assert paper.journal is None


class TestAuthorNode:
    """Test AuthorNode dataclass"""

    def test_create_author_node(self):
        """Test creating AuthorNode instance"""
        author = AuthorNode(
            name="Kim, John",
            affiliation="Harvard Medical School"
        )
        assert author.name == "Kim, John"
        assert author.affiliation == "Harvard Medical School"

    def test_create_author_node_minimal(self):
        """Test creating AuthorNode without affiliation"""
        author = AuthorNode(name="Smith, Jane")
        assert author.name == "Smith, Jane"
        assert author.affiliation is None


class TestKeywordNode:
    """Test KeywordNode dataclass"""

    def test_create_keyword_node(self):
        """Test creating KeywordNode instance"""
        keyword = KeywordNode(term="CRISPR", mesh_id="D000123")
        assert keyword.term == "CRISPR"
        assert keyword.mesh_id == "D000123"

    def test_create_keyword_node_minimal(self):
        """Test creating KeywordNode without mesh_id"""
        keyword = KeywordNode(term="Gene Therapy")
        assert keyword.term == "Gene Therapy"
        assert keyword.mesh_id is None


class TestCitationRelation:
    """Test CitationRelation dataclass"""

    def test_create_citation_relation(self):
        """Test creating CitationRelation instance"""
        citation = CitationRelation(
            citing_pmid="12345678",
            cited_pmid="87654321",
            context="As shown in previous work..."
        )
        assert citation.citing_pmid == "12345678"
        assert citation.cited_pmid == "87654321"
        assert citation.context == "As shown in previous work..."

    def test_create_citation_relation_no_context(self):
        """Test creating CitationRelation without context"""
        citation = CitationRelation(
            citing_pmid="11111111",
            cited_pmid="22222222"
        )
        assert citation.context is None


class TestGraphSearchResult:
    """Test GraphSearchResult dataclass"""

    def test_create_graph_search_result(self):
        """Test creating GraphSearchResult instance"""
        result = GraphSearchResult(
            pmid="12345678",
            title="Related Paper",
            relevance_score=0.85,
            path_type="citation",
            path_length=2,
            related_entities=[{"type": "keyword", "term": "CRISPR"}]
        )
        assert result.pmid == "12345678"
        assert result.relevance_score == 0.85
        assert result.path_type == "citation"
        assert len(result.related_entities) == 1


# ==================== GraphDBService Tests ====================

class TestGraphDBServiceInit:
    """Test GraphDBService initialization"""

    def test_init_with_defaults(self):
        """Test initialization with default values"""
        with patch('src.services.graph_db.settings') as mock_settings:
            mock_settings.NEO4J_URI = 'bolt://localhost:7687'
            mock_settings.NEO4J_USER = 'neo4j'
            mock_settings.NEO4J_PASSWORD = 'password'

            service = GraphDBService()

            assert service.uri == 'bolt://localhost:7687'
            assert service.user == 'neo4j'
            assert service._driver is None

    def test_init_with_custom_values(self):
        """Test initialization with custom values"""
        service = GraphDBService(
            uri="bolt://custom:7687",
            user="custom_user",
            password="custom_pass"
        )
        assert service.uri == "bolt://custom:7687"
        assert service.user == "custom_user"


class TestGraphDBServiceDriver:
    """Test Neo4j driver management"""

    @pytest.fixture
    def graph_service(self):
        return GraphDBService(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test"
        )

    def test_get_driver_creates_driver(self, graph_service):
        """Test _get_driver creates driver on first call"""
        mock_driver = MagicMock()

        with patch('src.services.graph_db.GraphDatabase.driver', return_value=mock_driver):
            driver = graph_service._get_driver()

            assert driver == mock_driver
            mock_driver.verify_connectivity.assert_called_once()

    def test_get_driver_connection_failure(self, graph_service):
        """Test _get_driver handles connection failure"""
        from neo4j.exceptions import ServiceUnavailable

        with patch('src.services.graph_db.GraphDatabase.driver') as mock:
            mock.side_effect = ServiceUnavailable("Connection failed")
            driver = graph_service._get_driver()

            assert driver is None

    def test_close_driver(self, graph_service):
        """Test close() properly closes driver"""
        mock_driver = MagicMock()
        graph_service._driver = mock_driver

        graph_service.close()

        mock_driver.close.assert_called_once()
        assert graph_service._driver is None

    def test_close_without_driver(self, graph_service):
        """Test close() when no driver exists"""
        graph_service.close()  # Should not raise


# ==================== Paper Operations Tests ====================

class TestPaperOperations:
    """Test paper CRUD operations"""

    @pytest.fixture
    def graph_service(self):
        return GraphDBService()

    @pytest.fixture
    def mock_driver_and_session(self):
        """Create mock driver and session"""
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        return mock_driver, mock_session

    def test_create_paper_success(self, graph_service, mock_driver_and_session):
        """Test successful paper creation"""
        mock_driver, mock_session = mock_driver_and_session
        mock_result = MagicMock()
        mock_result.single.return_value = {"pmid": "12345678"}
        mock_session.run.return_value = mock_result

        with patch.object(graph_service, '_get_driver', return_value=mock_driver):
            paper = PaperNode(
                pmid="12345678",
                title="Test Paper",
                abstract="Abstract",
                journal="Nature"
            )
            result = graph_service.create_paper(paper)

            assert result is True
            mock_session.run.assert_called_once()

    def test_create_paper_no_driver(self, graph_service):
        """Test create_paper returns False when no driver"""
        with patch.object(graph_service, '_get_driver', return_value=None):
            paper = PaperNode(pmid="12345678", title="Test")
            result = graph_service.create_paper(paper)

            assert result is False

    def test_create_papers_batch(self, graph_service, mock_driver_and_session):
        """Test batch paper creation"""
        mock_driver, mock_session = mock_driver_and_session
        mock_result = MagicMock()
        mock_result.single.return_value = {'count': 3}
        mock_session.run.return_value = mock_result

        with patch.object(graph_service, '_get_driver', return_value=mock_driver):
            papers = [
                PaperNode(pmid="1", title="Paper 1"),
                PaperNode(pmid="2", title="Paper 2"),
                PaperNode(pmid="3", title="Paper 3"),
            ]
            count = graph_service.create_papers_batch(papers)

            assert count == 3

    def test_create_papers_batch_no_driver(self, graph_service):
        """Test batch creation returns 0 when no driver"""
        with patch.object(graph_service, '_get_driver', return_value=None):
            papers = [PaperNode(pmid="1", title="Paper 1")]
            count = graph_service.create_papers_batch(papers)

            assert count == 0


# ==================== Citation Operations Tests ====================

class TestCitationOperations:
    """Test citation relationship operations"""

    @pytest.fixture
    def graph_service(self):
        return GraphDBService()

    @pytest.fixture
    def mock_driver_and_session(self):
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        return mock_driver, mock_session

    def test_create_citation_success(self, graph_service, mock_driver_and_session):
        """Test successful citation creation"""
        mock_driver, mock_session = mock_driver_and_session
        mock_result = MagicMock()
        mock_result.single.return_value = {"r": "relationship"}
        mock_session.run.return_value = mock_result

        with patch.object(graph_service, '_get_driver', return_value=mock_driver):
            citation = CitationRelation(
                citing_pmid="12345678",
                cited_pmid="87654321",
                context="Referenced work"
            )
            result = graph_service.create_citation(citation)

            assert result is True

    def test_create_citation_no_driver(self, graph_service):
        """Test create_citation returns False when no driver"""
        with patch.object(graph_service, '_get_driver', return_value=None):
            citation = CitationRelation(citing_pmid="1", cited_pmid="2")
            result = graph_service.create_citation(citation)

            assert result is False

    def test_create_citations_batch(self, graph_service, mock_driver_and_session):
        """Test batch citation creation"""
        mock_driver, mock_session = mock_driver_and_session
        mock_result = MagicMock()
        mock_result.single.return_value = {'count': 2}
        mock_session.run.return_value = mock_result

        with patch.object(graph_service, '_get_driver', return_value=mock_driver):
            citations = [
                CitationRelation(citing_pmid="1", cited_pmid="2"),
                CitationRelation(citing_pmid="2", cited_pmid="3"),
            ]
            count = graph_service.create_citations_batch(citations)

            assert count == 2


# ==================== Link Operations Tests ====================

class TestLinkOperations:
    """Test author and keyword linking"""

    @pytest.fixture
    def graph_service(self):
        return GraphDBService()

    @pytest.fixture
    def mock_driver_and_session(self):
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        return mock_driver, mock_session

    def test_link_author_to_paper(self, graph_service, mock_driver_and_session):
        """Test linking author to paper"""
        mock_driver, mock_session = mock_driver_and_session
        mock_result = MagicMock()
        mock_result.single.return_value = {"r": "relationship"}
        mock_session.run.return_value = mock_result

        with patch.object(graph_service, '_get_driver', return_value=mock_driver):
            result = graph_service.link_author_to_paper(
                author_name="Kim, John",
                pmid="12345678",
                position=1
            )

            assert result is True

    def test_link_author_no_driver(self, graph_service):
        """Test link_author returns False when no driver"""
        with patch.object(graph_service, '_get_driver', return_value=None):
            result = graph_service.link_author_to_paper("Kim", "12345678")

            assert result is False

    def test_link_keyword_to_paper(self, graph_service, mock_driver_and_session):
        """Test linking keyword to paper"""
        mock_driver, mock_session = mock_driver_and_session
        mock_result = MagicMock()
        mock_result.single.return_value = {"r": "relationship"}
        mock_session.run.return_value = mock_result

        with patch.object(graph_service, '_get_driver', return_value=mock_driver):
            result = graph_service.link_keyword_to_paper(
                keyword="CRISPR",
                pmid="12345678",
                mesh_id="D000123"
            )

            assert result is True

    def test_link_keyword_no_driver(self, graph_service):
        """Test link_keyword returns False when no driver"""
        with patch.object(graph_service, '_get_driver', return_value=None):
            result = graph_service.link_keyword_to_paper("CRISPR", "12345678")

            assert result is False


# ==================== Search Operations Tests ====================

class TestSearchOperations:
    """Test graph search operations"""

    @pytest.fixture
    def graph_service(self):
        return GraphDBService()

    @pytest.fixture
    def mock_driver_and_session(self):
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        return mock_driver, mock_session

    def test_find_citing_papers(self, graph_service, mock_driver_and_session):
        """Test finding papers that cite a paper"""
        mock_driver, mock_session = mock_driver_and_session
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([
            {"pmid": "111", "title": "Citing Paper 1", "journal": "Nature"},
            {"pmid": "222", "title": "Citing Paper 2", "journal": "Science"},
        ]))
        mock_session.run.return_value = mock_result

        with patch.object(graph_service, '_get_driver', return_value=mock_driver):
            results = graph_service.find_citing_papers("12345678", limit=10)

            assert len(results) == 2
            assert results[0]["pmid"] == "111"

    def test_find_citing_papers_no_driver(self, graph_service):
        """Test find_citing_papers returns empty when no driver"""
        with patch.object(graph_service, '_get_driver', return_value=None):
            results = graph_service.find_citing_papers("12345678")

            assert results == []

    def test_find_cited_papers(self, graph_service, mock_driver_and_session):
        """Test finding papers cited by a paper"""
        mock_driver, mock_session = mock_driver_and_session
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([
            {"pmid": "333", "title": "Cited Paper", "journal": "Cell"},
        ]))
        mock_session.run.return_value = mock_result

        with patch.object(graph_service, '_get_driver', return_value=mock_driver):
            results = graph_service.find_cited_papers("12345678")

            assert len(results) == 1

    def test_find_co_cited_papers(self, graph_service, mock_driver_and_session):
        """Test finding co-cited papers"""
        mock_driver, mock_session = mock_driver_and_session
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([
            {"pmid": "444", "title": "Co-cited Paper", "co_citation_count": 5},
        ]))
        mock_session.run.return_value = mock_result

        with patch.object(graph_service, '_get_driver', return_value=mock_driver):
            results = graph_service.find_co_cited_papers("12345678")

            assert len(results) == 1
            assert results[0]["co_citation_count"] == 5

    def test_find_related_by_keywords(self, graph_service, mock_driver_and_session):
        """Test finding papers by shared keywords"""
        mock_driver, mock_session = mock_driver_and_session
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([
            {"pmid": "555", "title": "Related", "shared_keywords": ["CRISPR", "Gene"], "keyword_count": 2},
        ]))
        mock_session.run.return_value = mock_result

        with patch.object(graph_service, '_get_driver', return_value=mock_driver):
            results = graph_service.find_related_by_keywords("12345678")

            assert len(results) == 1
            assert "CRISPR" in results[0]["shared_keywords"]

    def test_find_author_network(self, graph_service, mock_driver_and_session):
        """Test finding co-author network"""
        mock_driver, mock_session = mock_driver_and_session
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([
            {"co_author": "Park, Sarah", "collaborations": ["Paper 1"], "collab_count": 3},
        ]))
        mock_session.run.return_value = mock_result

        with patch.object(graph_service, '_get_driver', return_value=mock_driver):
            results = graph_service.find_author_network("Kim, John")

            assert len(results) == 1
            assert results[0]["co_author"] == "Park, Sarah"

    def test_find_citation_path(self, graph_service, mock_driver_and_session):
        """Test finding citation path between papers"""
        mock_driver, mock_session = mock_driver_and_session
        mock_result = MagicMock()
        mock_result.single.return_value = {
            'path': [
                {"pmid": "111", "title": "Start"},
                {"pmid": "222", "title": "Middle"},
                {"pmid": "333", "title": "End"},
            ],
            'path_length': 2
        }
        mock_session.run.return_value = mock_result

        with patch.object(graph_service, '_get_driver', return_value=mock_driver):
            path = graph_service.find_citation_path("111", "333")

            assert len(path) == 3

    def test_find_citation_path_no_path(self, graph_service, mock_driver_and_session):
        """Test find_citation_path when no path exists"""
        mock_driver, mock_session = mock_driver_and_session
        mock_result = MagicMock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result

        with patch.object(graph_service, '_get_driver', return_value=mock_driver):
            path = graph_service.find_citation_path("111", "999")

            assert path == []


# ==================== Korean Translation Tests ====================

class TestKoreanTranslation:
    """Test Korean medical term translation"""

    @pytest.fixture
    def graph_service(self):
        return GraphDBService()

    def test_translate_korean_cancer_terms(self, graph_service):
        """Test translation of Korean cancer terms"""
        assert graph_service._translate_korean_query("췌장암") == "pancreatic cancer"
        assert graph_service._translate_korean_query("폐암") == "lung cancer"
        assert graph_service._translate_korean_query("유방암") == "breast cancer"

    def test_translate_korean_treatment_terms(self, graph_service):
        """Test translation of Korean treatment terms"""
        assert graph_service._translate_korean_query("면역요법") == "immunotherapy"
        assert graph_service._translate_korean_query("유전자치료") == "gene therapy"

    def test_translate_korean_tech_terms(self, graph_service):
        """Test translation of Korean technology terms"""
        assert graph_service._translate_korean_query("크리스퍼") == "CRISPR"
        assert graph_service._translate_korean_query("딥러닝") == "deep learning"
        assert graph_service._translate_korean_query("인공지능") == "artificial intelligence"
        assert graph_service._translate_korean_query("머신러닝") == "machine learning"

    def test_translate_english_passthrough(self, graph_service):
        """Test that English queries pass through unchanged"""
        assert graph_service._translate_korean_query("CRISPR") == "CRISPR"
        assert graph_service._translate_korean_query("cancer therapy") == "cancer therapy"

    def test_translate_empty_query(self, graph_service):
        """Test translation of empty query"""
        assert graph_service._translate_korean_query("") == ""
        assert graph_service._translate_korean_query(None) is None

    def test_translate_unknown_korean(self, graph_service):
        """Test translation of unknown Korean term"""
        unknown = "알수없는용어"
        assert graph_service._translate_korean_query(unknown) == unknown


# ==================== Visualization Tests ====================

class TestVisualization:
    """Test visualization data generation"""

    @pytest.fixture
    def graph_service(self):
        return GraphDBService()

    @pytest.fixture
    def mock_driver_and_session(self):
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        return mock_driver, mock_session

    def test_get_visualization_data_no_driver(self, graph_service):
        """Test visualization returns disconnected status when no driver"""
        with patch.object(graph_service, '_get_driver', return_value=None):
            result = graph_service.get_visualization_data()

            assert result['status'] == 'disconnected'
            assert result['nodes'] == []
            assert result['edges'] == []

    def test_get_visualization_data_success(self, graph_service, mock_driver_and_session):
        """Test successful visualization data retrieval"""
        mock_driver, mock_session = mock_driver_and_session

        # Mock node results
        node_result = MagicMock()
        node_result.single.return_value = {
            'nodes': [
                {'id': 'paper_123', 'type': 'paper', 'label': 'Paper 1'},
                {'id': 'author_Kim', 'type': 'author', 'label': 'Kim'},
            ]
        }

        # Mock edge results
        edge_result = MagicMock()
        edge_result.single.return_value = {
            'edges': [
                {'source': 'author_Kim', 'target': 'paper_123', 'type': 'authored'},
            ]
        }

        mock_session.run.side_effect = [node_result, edge_result]

        with patch.object(graph_service, '_get_driver', return_value=mock_driver):
            result = graph_service.get_visualization_data(limit=50)

            assert result['status'] == 'connected'
            assert len(result['nodes']) == 2
            assert len(result['edges']) == 1

    def test_get_visualization_with_korean_query(self, graph_service, mock_driver_and_session):
        """Test visualization with Korean query translation"""
        mock_driver, mock_session = mock_driver_and_session

        node_result = MagicMock()
        node_result.single.return_value = {'nodes': []}
        edge_result = MagicMock()
        edge_result.single.return_value = {'edges': []}
        mock_session.run.side_effect = [node_result, edge_result]

        with patch.object(graph_service, '_get_driver', return_value=mock_driver):
            # Should translate 폐암 to "lung cancer"
            result = graph_service.get_visualization_data(query="폐암")

            assert result['status'] == 'connected'


# ==================== Statistics Tests ====================

class TestStatistics:
    """Test graph statistics"""

    @pytest.fixture
    def graph_service(self):
        return GraphDBService()

    @pytest.fixture
    def mock_driver_and_session(self):
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        return mock_driver, mock_session

    def test_get_stats_no_driver(self, graph_service):
        """Test get_stats returns disconnected when no driver"""
        with patch.object(graph_service, '_get_driver', return_value=None):
            stats = graph_service.get_stats()

            assert stats['status'] == 'disconnected'

    def test_get_stats_success(self, graph_service, mock_driver_and_session):
        """Test successful stats retrieval"""
        mock_driver, mock_session = mock_driver_and_session
        mock_result = MagicMock()
        mock_result.single.return_value = {
            'papers': 100,
            'authors': 250,
            'keywords': 50,
            'citations': 500,
            'authorships': 300,
            'keyword_links': 200
        }
        mock_session.run.return_value = mock_result

        with patch.object(graph_service, '_get_driver', return_value=mock_driver):
            stats = graph_service.get_stats()

            assert stats['status'] == 'connected'
            assert stats['papers'] == 100
            assert stats['citations'] == 500

    def test_get_stats_empty(self, graph_service, mock_driver_and_session):
        """Test get_stats with empty database"""
        mock_driver, mock_session = mock_driver_and_session
        mock_result = MagicMock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result

        with patch.object(graph_service, '_get_driver', return_value=mock_driver):
            stats = graph_service.get_stats()

            assert stats['status'] == 'empty'


# ==================== Singleton Tests ====================

class TestGetGraphService:
    """Test get_graph_service function"""

    def test_get_graph_service_singleton(self):
        """Test that get_graph_service returns singleton instance"""
        import src.services.graph_db as graph_db_module
        graph_db_module._graph_service = None

        service1 = get_graph_service()
        service2 = get_graph_service()

        assert service1 is service2
        assert isinstance(service1, GraphDBService)


# ==================== Schema Tests ====================

class TestSchemaSetup:
    """Test schema setup operations"""

    @pytest.fixture
    def graph_service(self):
        return GraphDBService()

    @pytest.fixture
    def mock_driver_and_session(self):
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        return mock_driver, mock_session

    def test_setup_schema_no_driver(self, graph_service):
        """Test schema setup when no driver available"""
        with patch.object(graph_service, '_get_driver', return_value=None):
            graph_service.setup_schema()  # Should not raise

    def test_setup_schema_success(self, graph_service, mock_driver_and_session):
        """Test successful schema setup"""
        mock_driver, mock_session = mock_driver_and_session
        mock_session.run.return_value = MagicMock()

        with patch.object(graph_service, '_get_driver', return_value=mock_driver):
            graph_service.setup_schema()

            # Should have called run multiple times for constraints and indexes
            assert mock_session.run.call_count >= 6  # 3 constraints + 3 indexes
