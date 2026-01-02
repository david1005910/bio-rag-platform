"""Tests for PubMed Service"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.pubmed import PubMedService, PubMedPaper, get_pubmed_service


# Sample XML response for testing
SAMPLE_PUBMED_XML = """<?xml version="1.0" ?>
<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2024//EN" "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_240101.dtd">
<PubmedArticleSet>
    <PubmedArticle>
        <MedlineCitation Status="MEDLINE" Owner="NLM">
            <PMID Version="1">12345678</PMID>
            <Article PubModel="Print">
                <Journal>
                    <Title>Nature Medicine</Title>
                    <JournalIssue>
                        <PubDate>
                            <Year>2024</Year>
                            <Month>01</Month>
                            <Day>15</Day>
                        </PubDate>
                    </JournalIssue>
                </Journal>
                <ArticleTitle>CRISPR-Cas9 Gene Therapy for Cancer Treatment</ArticleTitle>
                <Abstract>
                    <AbstractText Label="BACKGROUND">Cancer remains a major health challenge.</AbstractText>
                    <AbstractText Label="METHODS">We developed a novel CRISPR approach.</AbstractText>
                    <AbstractText Label="RESULTS">Significant tumor reduction was observed.</AbstractText>
                </Abstract>
                <AuthorList>
                    <Author>
                        <LastName>Kim</LastName>
                        <ForeName>John</ForeName>
                    </Author>
                    <Author>
                        <LastName>Park</LastName>
                        <ForeName>Sarah</ForeName>
                    </Author>
                </AuthorList>
                <ELocationID EIdType="doi">10.1038/nm.2024.001</ELocationID>
            </Article>
            <KeywordList>
                <Keyword>CRISPR</Keyword>
                <Keyword>Gene Therapy</Keyword>
                <Keyword>Cancer</Keyword>
            </KeywordList>
            <MeshHeadingList>
                <MeshHeading>
                    <DescriptorName>Neoplasms</DescriptorName>
                </MeshHeading>
                <MeshHeading>
                    <DescriptorName>CRISPR-Cas Systems</DescriptorName>
                </MeshHeading>
            </MeshHeadingList>
        </MedlineCitation>
    </PubmedArticle>
</PubmedArticleSet>
"""

SAMPLE_SEARCH_RESPONSE = {
    "esearchresult": {
        "count": "1500",
        "idlist": ["12345678", "87654321", "11111111"]
    }
}


class AsyncContextManager:
    """Helper class to mock async context managers"""

    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class TestPubMedPaper:
    """Test PubMedPaper dataclass"""

    def test_create_paper(self):
        """Test creating PubMedPaper instance"""
        paper = PubMedPaper(
            pmid="12345678",
            title="Test Paper Title",
            abstract="Test abstract content",
            authors=["Kim, John", "Park, Sarah"],
            journal="Nature Medicine",
            publication_date="2024-01-15",
            doi="10.1038/test",
            keywords=["CRISPR", "Cancer"],
            mesh_terms=["Neoplasms"]
        )
        assert paper.pmid == "12345678"
        assert paper.title == "Test Paper Title"
        assert len(paper.authors) == 2
        assert paper.doi == "10.1038/test"

    def test_create_paper_with_optional_fields(self):
        """Test creating PubMedPaper with optional fields as None"""
        paper = PubMedPaper(
            pmid="12345678",
            title="Test Paper",
            abstract="Abstract",
            authors=[],
            journal="Test Journal",
            publication_date=None,
            doi=None,
            keywords=[],
            mesh_terms=[]
        )
        assert paper.pmid == "12345678"
        assert paper.publication_date is None
        assert paper.doi is None


class TestPubMedService:
    """Test PubMedService class"""

    @pytest.fixture
    def pubmed_service(self):
        """Create PubMedService instance for testing"""
        return PubMedService(api_key="test_key", email="test@example.com")

    def test_init_with_credentials(self):
        """Test initialization with API key and email"""
        service = PubMedService(api_key="my_key", email="my@email.com")
        assert service.api_key == "my_key"
        assert service.email == "my@email.com"

    def test_init_without_credentials(self):
        """Test initialization without credentials"""
        service = PubMedService()
        assert service.api_key == ""
        assert service.email == ""

    def test_build_params_with_credentials(self, pubmed_service):
        """Test _build_params adds API key and email"""
        base_params = {"db": "pubmed", "term": "cancer"}
        result = pubmed_service._build_params(base_params)

        assert result["db"] == "pubmed"
        assert result["term"] == "cancer"
        assert result["api_key"] == "test_key"
        assert result["email"] == "test@example.com"

    def test_build_params_without_credentials(self):
        """Test _build_params without credentials"""
        service = PubMedService()
        base_params = {"db": "pubmed"}
        result = service._build_params(base_params)

        assert result["db"] == "pubmed"
        assert "api_key" not in result
        assert "email" not in result

    @pytest.mark.asyncio
    async def test_search_success(self, pubmed_service):
        """Test successful search"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SAMPLE_SEARCH_RESPONSE)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncContextManager(mock_response))

        with patch.object(pubmed_service, '_get_session', return_value=mock_session):
            total, pmids = await pubmed_service.search("cancer", max_results=10)

        assert total == 1500
        assert len(pmids) == 3
        assert "12345678" in pmids

    @pytest.mark.asyncio
    async def test_search_with_date_filters(self, pubmed_service):
        """Test search with date filters"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SAMPLE_SEARCH_RESPONSE)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncContextManager(mock_response))

        with patch.object(pubmed_service, '_get_session', return_value=mock_session):
            total, pmids = await pubmed_service.search(
                "cancer",
                min_date="2020/01/01",
                max_date="2024/12/31"
            )

        assert total == 1500
        # Verify date params were passed
        call_args = mock_session.get.call_args
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_search_api_error(self, pubmed_service):
        """Test search when API returns error"""
        mock_response = MagicMock()
        mock_response.status = 500

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncContextManager(mock_response))

        with patch.object(pubmed_service, '_get_session', return_value=mock_session):
            total, pmids = await pubmed_service.search("cancer")

        assert total == 0
        assert pmids == []

    @pytest.mark.asyncio
    async def test_search_exception(self, pubmed_service):
        """Test search when exception occurs"""
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=Exception("Network error"))

        with patch.object(pubmed_service, '_get_session', return_value=mock_session):
            total, pmids = await pubmed_service.search("cancer")

        assert total == 0
        assert pmids == []

    @pytest.mark.asyncio
    async def test_fetch_papers_empty_list(self, pubmed_service):
        """Test fetch_papers with empty PMID list"""
        result = await pubmed_service.fetch_papers([])
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_papers_success(self, pubmed_service):
        """Test successful paper fetch"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=SAMPLE_PUBMED_XML)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncContextManager(mock_response))

        with patch.object(pubmed_service, '_get_session', return_value=mock_session):
            papers = await pubmed_service.fetch_papers(["12345678"])

        assert len(papers) == 1
        paper = papers[0]
        assert paper.pmid == "12345678"
        assert "CRISPR" in paper.title
        assert paper.journal == "Nature Medicine"
        assert len(paper.authors) == 2
        assert paper.authors[0] == "Kim, John"
        assert paper.doi == "10.1038/nm.2024.001"
        assert "CRISPR" in paper.keywords
        assert "Neoplasms" in paper.mesh_terms

    @pytest.mark.asyncio
    async def test_fetch_papers_api_error(self, pubmed_service):
        """Test fetch_papers when API returns error"""
        mock_response = MagicMock()
        mock_response.status = 500

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncContextManager(mock_response))

        with patch.object(pubmed_service, '_get_session', return_value=mock_session):
            papers = await pubmed_service.fetch_papers(["12345678"])

        assert papers == []

    def test_parse_pubmed_xml(self, pubmed_service):
        """Test XML parsing"""
        papers = pubmed_service._parse_pubmed_xml(SAMPLE_PUBMED_XML)

        assert len(papers) == 1
        paper = papers[0]
        assert paper.pmid == "12345678"
        assert "CRISPR-Cas9" in paper.title
        assert "BACKGROUND:" in paper.abstract
        assert "METHODS:" in paper.abstract
        assert "RESULTS:" in paper.abstract

    def test_parse_pubmed_xml_invalid(self, pubmed_service):
        """Test XML parsing with invalid XML"""
        papers = pubmed_service._parse_pubmed_xml("<invalid>xml")
        assert papers == []

    def test_parse_pubmed_xml_empty(self, pubmed_service):
        """Test XML parsing with empty article set"""
        xml = """<?xml version="1.0" ?>
        <PubmedArticleSet></PubmedArticleSet>"""
        papers = pubmed_service._parse_pubmed_xml(xml)
        assert papers == []

    def test_get_text_content(self, pubmed_service):
        """Test _get_text_content extracts nested text"""
        import xml.etree.ElementTree as ET
        elem = ET.fromstring("<title>Main <i>italic</i> text</title>")
        result = pubmed_service._get_text_content(elem)
        assert result == "Main italic text"

    @pytest.mark.asyncio
    async def test_search_and_fetch_success(self, pubmed_service):
        """Test combined search and fetch"""
        mock_search_response = MagicMock()
        mock_search_response.status = 200
        mock_search_response.json = AsyncMock(return_value=SAMPLE_SEARCH_RESPONSE)

        mock_fetch_response = MagicMock()
        mock_fetch_response.status = 200
        mock_fetch_response.text = AsyncMock(return_value=SAMPLE_PUBMED_XML)

        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AsyncContextManager(mock_search_response)
            else:
                return AsyncContextManager(mock_fetch_response)

        mock_session = MagicMock()
        mock_session.get = mock_get

        with patch.object(pubmed_service, '_get_session', return_value=mock_session):
            total, papers = await pubmed_service.search_and_fetch("cancer", max_results=10)

        assert total == 1500
        assert len(papers) >= 1

    @pytest.mark.asyncio
    async def test_search_and_fetch_with_filters(self, pubmed_service):
        """Test search_and_fetch with year and journal filters"""
        mock_search_response = MagicMock()
        mock_search_response.status = 200
        mock_search_response.json = AsyncMock(return_value={
            "esearchresult": {"count": "100", "idlist": ["12345678"]}
        })

        mock_fetch_response = MagicMock()
        mock_fetch_response.status = 200
        mock_fetch_response.text = AsyncMock(return_value=SAMPLE_PUBMED_XML)

        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AsyncContextManager(mock_search_response)
            else:
                return AsyncContextManager(mock_fetch_response)

        mock_session = MagicMock()
        mock_session.get = mock_get

        with patch.object(pubmed_service, '_get_session', return_value=mock_session):
            total, papers = await pubmed_service.search_and_fetch(
                "cancer",
                year_from=2020,
                year_to=2024,
                journals=["Nature", "Science"],
                authors=["Kim"]
            )

        assert total == 100
        assert len(papers) == 1

    @pytest.mark.asyncio
    async def test_search_and_fetch_no_results(self, pubmed_service):
        """Test search_and_fetch when no results found"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "esearchresult": {"count": "0", "idlist": []}
        })

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncContextManager(mock_response))

        with patch.object(pubmed_service, '_get_session', return_value=mock_session):
            total, papers = await pubmed_service.search_and_fetch("xyznonexistent")

        assert total == 0
        assert papers == []

    @pytest.mark.asyncio
    async def test_close_session(self, pubmed_service):
        """Test closing the session"""
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()

        pubmed_service._session = mock_session
        await pubmed_service.close()

        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_session_already_closed(self, pubmed_service):
        """Test closing already closed session"""
        mock_session = MagicMock()
        mock_session.closed = True

        pubmed_service._session = mock_session
        await pubmed_service.close()  # Should not raise


class TestGetPubMedService:
    """Test get_pubmed_service function"""

    def test_get_pubmed_service_singleton(self):
        """Test that get_pubmed_service returns singleton instance"""
        # Reset the global instance
        import src.services.pubmed as pubmed_module
        pubmed_module._pubmed_service = None

        service1 = get_pubmed_service(api_key="key1", email="email1@test.com")
        service2 = get_pubmed_service(api_key="key2", email="email2@test.com")

        # Should return same instance
        assert service1 is service2
        # First call's credentials should be used
        assert service1.api_key == "key1"
        assert isinstance(service1, PubMedService)


class TestPubMedXMLParsing:
    """Test edge cases in XML parsing"""

    @pytest.fixture
    def pubmed_service(self):
        return PubMedService()

    def test_parse_article_without_abstract(self, pubmed_service):
        """Test parsing article without abstract"""
        xml = """<?xml version="1.0" ?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>99999999</PMID>
                    <Article>
                        <Journal><Title>Test Journal</Title></Journal>
                        <ArticleTitle>No Abstract Paper</ArticleTitle>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        papers = pubmed_service._parse_pubmed_xml(xml)
        assert len(papers) == 1
        assert papers[0].abstract == ""

    def test_parse_article_without_authors(self, pubmed_service):
        """Test parsing article without author list"""
        xml = """<?xml version="1.0" ?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>99999999</PMID>
                    <Article>
                        <Journal><Title>Test Journal</Title></Journal>
                        <ArticleTitle>No Authors Paper</ArticleTitle>
                        <Abstract>
                            <AbstractText>Test abstract</AbstractText>
                        </Abstract>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        papers = pubmed_service._parse_pubmed_xml(xml)
        assert len(papers) == 1
        assert papers[0].authors == []

    def test_parse_article_with_month_name(self, pubmed_service):
        """Test parsing article with month as name (Jan, Feb, etc.)"""
        xml = """<?xml version="1.0" ?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>99999999</PMID>
                    <Article>
                        <Journal>
                            <Title>Test Journal</Title>
                            <JournalIssue>
                                <PubDate>
                                    <Year>2024</Year>
                                    <Month>Jan</Month>
                                </PubDate>
                            </JournalIssue>
                        </Journal>
                        <ArticleTitle>Test Paper</ArticleTitle>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        papers = pubmed_service._parse_pubmed_xml(xml)
        assert len(papers) == 1
        assert papers[0].publication_date == "2024-Jan"
