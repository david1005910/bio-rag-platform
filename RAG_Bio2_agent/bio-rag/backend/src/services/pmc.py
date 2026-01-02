"""PMC (PubMed Central) Service - Free Full-Text PDF Access"""

import aiohttp
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# NCBI API endpoints (updated URLs)
ID_CONVERTER_URL = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
PMC_OA_URL = "https://pmc.ncbi.nlm.nih.gov/tools/oa/api/"


@dataclass
class PMCPaperInfo:
    """PMC paper information"""
    pmid: str
    pmcid: Optional[str]
    doi: Optional[str]
    has_pdf: bool
    pdf_url: Optional[str]
    is_open_access: bool


class PMCService:
    """Service for accessing PubMed Central resources"""

    def __init__(self, email: str = "bio-rag@example.com"):
        self.email = email
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            # Use a proper User-Agent to avoid 403 errors from PMC
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def close(self):
        """Close the session"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def convert_pmid_to_pmcid(self, pmids: List[str]) -> Dict[str, Optional[str]]:
        """
        Convert PMIDs to PMCIDs using NCBI ID Converter

        Args:
            pmids: List of PubMed IDs

        Returns:
            Dict mapping PMID to PMCID (None if not in PMC)
        """
        if not pmids:
            return {}

        session = await self._get_session()
        results = {}

        try:
            params = {
                "ids": ",".join(pmids),
                "format": "json",
                "tool": "bio-rag",
                "email": self.email
            }

            async with session.get(ID_CONVERTER_URL, params=params) as response:
                if response.status != 200:
                    logger.error(f"ID converter failed: {response.status}")
                    return {pmid: None for pmid in pmids}

                data = await response.json()

                # Check if the response is successful
                if data.get("status") != "ok":
                    logger.error(f"ID converter returned error status: {data}")
                    return {pmid: None for pmid in pmids}

                records = data.get("records", [])

                for record in records:
                    # Handle both string and integer pmid values
                    pmid_val = record.get("pmid", "")
                    pmid_str = str(pmid_val) if pmid_val else ""
                    pmcid = record.get("pmcid")
                    if pmid_str:
                        results[pmid_str] = pmcid
                        logger.info(f"Converted PMID {pmid_str} to PMCID {pmcid}")

                # Fill in missing PMIDs
                for pmid in pmids:
                    if pmid not in results:
                        results[pmid] = None

                return results

        except Exception as e:
            logger.error(f"PMID to PMCID conversion error: {e}")
            return {pmid: None for pmid in pmids}

    async def check_open_access(self, pmcid: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a PMC article is open access and get PDF URL

        Args:
            pmcid: PMC ID (e.g., "PMC1234567")

        Returns:
            Tuple of (is_open_access, pdf_url)
        """
        # PMC articles with a PMCID are typically open access
        # We can construct the PDF URL directly using the standard PMC URL pattern

        if pmcid:
            # New PMC URL pattern: https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/pdf/
            pdf_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/"

            # Verify the PDF is accessible
            session = await self._get_session()
            try:
                # Use HEAD request with allow_redirects=False to check the initial response
                async with session.head(pdf_url, allow_redirects=False) as response:
                    # 301/302 redirect means PDF is available
                    if response.status in (200, 301, 302):
                        # Get the redirected URL if it's a redirect
                        if response.status in (301, 302):
                            location = response.headers.get('Location', '')
                            if location:
                                # Construct full URL if relative
                                if location.startswith('/'):
                                    final_url = f"https://pmc.ncbi.nlm.nih.gov{location}"
                                else:
                                    final_url = location
                                logger.info(f"PDF found for {pmcid}: {final_url}")
                                return True, final_url
                        return True, pdf_url

                    # If not found at standard location, try with GET (some servers block HEAD)
                    async with session.get(pdf_url, allow_redirects=True) as get_response:
                        if get_response.status == 200:
                            final_url = str(get_response.url)
                            logger.info(f"PDF found for {pmcid}: {final_url}")
                            return True, final_url

            except Exception as e:
                logger.warning(f"PDF URL check failed for {pmcid}: {e}")
                # Still return the URL as it might work in browser
                return True, pdf_url

        return False, None

    async def get_pdf_info(self, pmids: List[str]) -> Dict[str, PMCPaperInfo]:
        """
        Get PDF availability info for multiple papers

        Args:
            pmids: List of PubMed IDs

        Returns:
            Dict mapping PMID to PMCPaperInfo
        """
        results = {}

        # First, convert PMIDs to PMCIDs
        pmcid_map = await self.convert_pmid_to_pmcid(pmids)

        for pmid in pmids:
            pmcid = pmcid_map.get(pmid)

            if pmcid:
                # Check if open access
                is_oa, pdf_url = await self.check_open_access(pmcid)

                # If no direct PDF URL from OA service, construct PMC PDF URL
                if is_oa and not pdf_url:
                    # Direct link to PMC article (may have PDF viewer)
                    pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"

                results[pmid] = PMCPaperInfo(
                    pmid=pmid,
                    pmcid=pmcid,
                    doi=None,
                    has_pdf=is_oa,
                    pdf_url=pdf_url,
                    is_open_access=is_oa
                )
            else:
                results[pmid] = PMCPaperInfo(
                    pmid=pmid,
                    pmcid=None,
                    doi=None,
                    has_pdf=False,
                    pdf_url=None,
                    is_open_access=False
                )

        return results

    async def get_single_pdf_info(self, pmid: str) -> PMCPaperInfo:
        """
        Get PDF info for a single paper

        Args:
            pmid: PubMed ID

        Returns:
            PMCPaperInfo
        """
        results = await self.get_pdf_info([pmid])
        return results.get(pmid, PMCPaperInfo(
            pmid=pmid,
            pmcid=None,
            doi=None,
            has_pdf=False,
            pdf_url=None,
            is_open_access=False
        ))

    async def download_pdf(self, pmid: str) -> Tuple[Optional[bytes], str]:
        """
        Download PDF for a paper if available

        Args:
            pmid: PubMed ID

        Returns:
            Tuple of (pdf_bytes, filename) or (None, error_message)
        """
        info = await self.get_single_pdf_info(pmid)

        if not info.has_pdf or not info.pdf_url:
            return None, f"PDF not available for PMID {pmid}"

        session = await self._get_session()

        try:
            async with session.get(info.pdf_url) as response:
                if response.status != 200:
                    return None, f"Failed to download PDF: {response.status}"

                content = await response.read()
                filename = f"{pmid}_{info.pmcid or 'paper'}.pdf"
                return content, filename

        except Exception as e:
            logger.error(f"PDF download error for {pmid}: {e}")
            return None, str(e)


# Global service instance
_pmc_service: Optional[PMCService] = None


def get_pmc_service() -> PMCService:
    """Get or create PMC service instance"""
    global _pmc_service
    if _pmc_service is None:
        _pmc_service = PMCService()
    return _pmc_service
