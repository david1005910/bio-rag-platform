"""Paper crawling tasks"""

import asyncio
import logging
from typing import List

from .celery import celery_app

logger = logging.getLogger(__name__)


async def _save_papers_to_db(papers: list) -> int:
    """Save papers to database and queue embedding generation"""
    from src.core.database import AsyncSessionLocal
    from src.services.paper_service import PaperService
    from src.tasks.embedding import process_paper_embeddings

    saved_count = 0

    async with AsyncSessionLocal() as db:
        paper_service = PaperService(db)

        for paper in papers:
            try:
                # Get or create paper in database
                db_paper, created = await paper_service.get_or_create(
                    pmid=paper.pmid,
                    title=paper.title,
                    abstract=paper.abstract,
                    authors=paper.authors if hasattr(paper, 'authors') else [],
                    journal=paper.journal if hasattr(paper, 'journal') else None,
                    publication_date=paper.publication_date if hasattr(paper, 'publication_date') else None,
                    doi=paper.doi if hasattr(paper, 'doi') else None,
                    keywords=paper.keywords if hasattr(paper, 'keywords') else [],
                    mesh_terms=paper.mesh_terms if hasattr(paper, 'mesh_terms') else [],
                )

                if created:
                    saved_count += 1
                    # Queue embedding generation for new papers
                    process_paper_embeddings.delay(str(db_paper.id))
                    logger.info(f"Queued embedding generation for paper: {paper.pmid}")

            except Exception as e:
                logger.error(f"Failed to save paper {paper.pmid}: {e}")

        await db.commit()

    return saved_count

# Default keywords to crawl
DEFAULT_KEYWORDS = [
    "CAR-T cell therapy",
    "CRISPR gene editing",
    "cancer immunotherapy",
    "mRNA vaccine",
    "gene therapy",
    "checkpoint inhibitor",
    "PD-1 PD-L1",
]


@celery_app.task(name="src.tasks.crawler.daily_paper_crawl")
def daily_paper_crawl(keywords: List[str] = None) -> dict:
    """
    Daily task to crawl new papers from PubMed.

    Args:
        keywords: List of search keywords (uses defaults if None)

    Returns:
        Summary of crawl results
    """
    keywords = keywords or DEFAULT_KEYWORDS

    # Run async crawl
    result = asyncio.run(_async_crawl(keywords))

    return result


async def _async_crawl(keywords: List[str]) -> dict:
    """Async implementation of paper crawling."""
    from src.services.collector.pubmed_collector import PubMedCollector

    collector = PubMedCollector()

    total_papers = 0
    results = {}

    for keyword in keywords:
        try:
            logger.info(f"Crawling papers for: {keyword}")

            # Search for recent papers (last 7 days)
            papers = await collector.search_and_fetch(
                query=f"{keyword}[Title/Abstract]",
                max_results=50
            )

            results[keyword] = len(papers)
            total_papers += len(papers)

            # Save papers to database and trigger embedding generation
            saved_count = await _save_papers_to_db(papers)
            logger.info(f"Saved {saved_count} new papers to database")

            logger.info(f"Found {len(papers)} papers for '{keyword}'")

        except Exception as e:
            logger.error(f"Error crawling '{keyword}': {e}")
            results[keyword] = 0

    return {
        "total_papers": total_papers,
        "by_keyword": results,
        "status": "completed"
    }


@celery_app.task(name="src.tasks.crawler.crawl_keyword")
def crawl_keyword(keyword: str, max_results: int = 100) -> dict:
    """
    Crawl papers for a specific keyword.

    Args:
        keyword: Search keyword
        max_results: Maximum papers to fetch

    Returns:
        Crawl results
    """
    result = asyncio.run(_async_crawl_keyword(keyword, max_results))
    return result


async def _async_crawl_keyword(keyword: str, max_results: int) -> dict:
    """Async implementation for single keyword crawl."""
    from src.services.collector.pubmed_collector import PubMedCollector

    collector = PubMedCollector()

    try:
        papers = await collector.search_and_fetch(
            query=f"{keyword}[Title/Abstract]",
            max_results=max_results
        )

        return {
            "keyword": keyword,
            "papers_found": len(papers),
            "status": "completed"
        }

    except Exception as e:
        logger.error(f"Error crawling '{keyword}': {e}")
        return {
            "keyword": keyword,
            "papers_found": 0,
            "status": "error",
            "error": str(e)
        }
