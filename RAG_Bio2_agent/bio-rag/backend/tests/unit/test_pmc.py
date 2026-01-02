"""Tests for PMC (PubMed Central) Service"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from aiohttp import ClientResponseError

from src.services.pmc import PMCService, PMCPaperInfo, get_pmc_service


class TestPMCPaperInfo:
    """Test PMCPaperInfo dataclass"""

    def test_create_paper_info_with_pdf(self):
        """Test creating PMCPaperInfo with PDF available"""
        info = PMCPaperInfo(
            pmid="12345678",
            pmcid="PMC1234567",
            doi="10.1234/test",
            has_pdf=True,
            pdf_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/pdf/",
            is_open_access=True
        )
        assert info.pmid == "12345678"
        assert info.pmcid == "PMC1234567"
        assert info.has_pdf is True
        assert info.is_open_access is True

    def test_create_paper_info_without_pdf(self):
        """Test creating PMCPaperInfo without PDF"""
        info = PMCPaperInfo(
            pmid="87654321",
            pmcid=None,
            doi=None,
            has_pdf=False,
            pdf_url=None,
            is_open_access=False
        )
        assert info.pmid == "87654321"
        assert info.pmcid is None
        assert info.has_pdf is False
        assert info.pdf_url is None


class TestPMCService:
    """Test PMCService class"""

    @pytest.fixture
    def pmc_service(self):
        """Create PMCService instance for testing"""
        return PMCService(email="test@example.com")

    @pytest.mark.asyncio
    async def test_convert_pmid_to_pmcid_empty_list(self, pmc_service):
        """Test conversion with empty PMID list"""
        result = await pmc_service.convert_pmid_to_pmcid([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_convert_pmid_to_pmcid_success(self, pmc_service):
        """Test successful PMID to PMCID conversion"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "status": "ok",
            "records": [
                {"pmid": "12345678", "pmcid": "PMC1234567"},
                {"pmid": "87654321", "pmcid": None}
            ]
        })

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncContextManager(mock_response))

        with patch.object(pmc_service, '_get_session', return_value=mock_session):
            result = await pmc_service.convert_pmid_to_pmcid(["12345678", "87654321"])

        assert result["12345678"] == "PMC1234567"
        assert result["87654321"] is None

    @pytest.mark.asyncio
    async def test_convert_pmid_to_pmcid_api_error(self, pmc_service):
        """Test PMID conversion when API returns error status"""
        mock_response = MagicMock()
        mock_response.status = 500

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncContextManager(mock_response))

        with patch.object(pmc_service, '_get_session', return_value=mock_session):
            result = await pmc_service.convert_pmid_to_pmcid(["12345678"])

        assert result["12345678"] is None

    @pytest.mark.asyncio
    async def test_check_open_access_with_valid_pmcid(self, pmc_service):
        """Test open access check with valid PMCID"""
        mock_head_response = MagicMock()
        mock_head_response.status = 302
        mock_head_response.headers = {"Location": "/articles/PMC1234567/pdf/main.pdf"}

        mock_session = MagicMock()
        mock_session.head = MagicMock(return_value=AsyncContextManager(mock_head_response))

        with patch.object(pmc_service, '_get_session', return_value=mock_session):
            is_oa, pdf_url = await pmc_service.check_open_access("PMC1234567")

        assert is_oa is True
        assert pdf_url == "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/pdf/main.pdf"

    @pytest.mark.asyncio
    async def test_check_open_access_with_empty_pmcid(self, pmc_service):
        """Test open access check with empty PMCID"""
        is_oa, pdf_url = await pmc_service.check_open_access("")
        assert is_oa is False
        assert pdf_url is None

    @pytest.mark.asyncio
    async def test_check_open_access_with_none_pmcid(self, pmc_service):
        """Test open access check with None PMCID"""
        is_oa, pdf_url = await pmc_service.check_open_access(None)
        assert is_oa is False
        assert pdf_url is None

    @pytest.mark.asyncio
    async def test_get_single_pdf_info(self, pmc_service):
        """Test getting PDF info for a single paper"""
        mock_pdf_info = {
            "12345678": PMCPaperInfo(
                pmid="12345678",
                pmcid="PMC1234567",
                doi=None,
                has_pdf=True,
                pdf_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/pdf/",
                is_open_access=True
            )
        }

        with patch.object(pmc_service, 'get_pdf_info', return_value=mock_pdf_info):
            result = await pmc_service.get_single_pdf_info("12345678")

        assert result.pmid == "12345678"
        assert result.has_pdf is True

    @pytest.mark.asyncio
    async def test_get_single_pdf_info_not_found(self, pmc_service):
        """Test getting PDF info when paper not found"""
        with patch.object(pmc_service, 'get_pdf_info', return_value={}):
            result = await pmc_service.get_single_pdf_info("99999999")

        assert result.pmid == "99999999"
        assert result.has_pdf is False
        assert result.pmcid is None

    @pytest.mark.asyncio
    async def test_download_pdf_not_available(self, pmc_service):
        """Test PDF download when not available"""
        mock_info = PMCPaperInfo(
            pmid="12345678",
            pmcid=None,
            doi=None,
            has_pdf=False,
            pdf_url=None,
            is_open_access=False
        )

        with patch.object(pmc_service, 'get_single_pdf_info', return_value=mock_info):
            content, message = await pmc_service.download_pdf("12345678")

        assert content is None
        assert "not available" in message

    @pytest.mark.asyncio
    async def test_download_pdf_success(self, pmc_service):
        """Test successful PDF download"""
        mock_info = PMCPaperInfo(
            pmid="12345678",
            pmcid="PMC1234567",
            doi=None,
            has_pdf=True,
            pdf_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/pdf/",
            is_open_access=True
        )

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b"%PDF-1.4 test content")

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncContextManager(mock_response))

        with patch.object(pmc_service, 'get_single_pdf_info', return_value=mock_info):
            with patch.object(pmc_service, '_get_session', return_value=mock_session):
                content, filename = await pmc_service.download_pdf("12345678")

        assert content == b"%PDF-1.4 test content"
        assert "12345678" in filename
        assert "PMC1234567" in filename

    @pytest.mark.asyncio
    async def test_close_session(self, pmc_service):
        """Test closing the session"""
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()

        pmc_service._session = mock_session
        await pmc_service.close()

        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_session_already_closed(self, pmc_service):
        """Test closing already closed session"""
        mock_session = MagicMock()
        mock_session.closed = True

        pmc_service._session = mock_session
        await pmc_service.close()  # Should not raise


class TestGetPMCService:
    """Test get_pmc_service function"""

    def test_get_pmc_service_singleton(self):
        """Test that get_pmc_service returns singleton instance"""
        # Reset the global instance
        import src.services.pmc as pmc_module
        pmc_module._pmc_service = None

        service1 = get_pmc_service()
        service2 = get_pmc_service()

        assert service1 is service2
        assert isinstance(service1, PMCService)


class AsyncContextManager:
    """Helper class to mock async context managers"""

    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
