"""Embedding generation tasks"""

import asyncio
import logging
from typing import List
import uuid

from .celery import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="src.tasks.embedding.process_paper_embeddings")
def process_paper_embeddings(paper_id: str) -> dict:
    """
    Generate embeddings for a paper and store in vector DB.

    Args:
        paper_id: Paper ID (UUID)

    Returns:
        Processing result
    """
    result = asyncio.run(_async_process_paper(paper_id))
    return result


async def _async_process_paper(paper_id: str) -> dict:
    """Async implementation of paper embedding processing."""
    from uuid import UUID
    from src.services.embedding.generator import EmbeddingGenerator
    from src.services.embedding.chunker import TextChunker
    from src.services.storage.vector_store import VectorStore
    from src.core.database import AsyncSessionLocal
    from src.services.paper_service import PaperService

    try:
        # Fetch paper from database
        async with AsyncSessionLocal() as db:
            paper_service = PaperService(db)
            paper = await paper_service.get_by_id(UUID(paper_id))

            if not paper:
                return {
                    "paper_id": paper_id,
                    "status": "error",
                    "error": "Paper not found in database"
                }

            paper_data = {
                "id": str(paper.id),
                "pmid": paper.pmid,
                "title": paper.title,
                "abstract": paper.abstract or ""
            }

        # Initialize services
        chunker = TextChunker()
        embedding_generator = EmbeddingGenerator()
        vector_store = VectorStore()

        # Chunk the paper
        chunks = chunker.chunk_paper(
            title=paper_data["title"],
            abstract=paper_data["abstract"]
        )

        if not chunks:
            return {
                "paper_id": paper_id,
                "status": "skipped",
                "reason": "No chunks generated"
            }

        # Generate embeddings
        texts = [c.text for c in chunks]
        embeddings = embedding_generator.batch_encode(texts)

        # Prepare metadata
        metadatas = [
            {
                "pmid": paper_data["pmid"],
                "title": paper_data["title"],
                "section": c.section,
                "chunk_index": c.index
            }
            for c in chunks
        ]

        # Store in vector DB
        ids = [str(uuid.uuid4()) for _ in chunks]
        vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        logger.info(f"Processed {len(chunks)} chunks for paper {paper_id}")

        return {
            "paper_id": paper_id,
            "chunks_created": len(chunks),
            "status": "completed"
        }

    except Exception as e:
        logger.error(f"Error processing paper {paper_id}: {e}")
        return {
            "paper_id": paper_id,
            "status": "error",
            "error": str(e)
        }


@celery_app.task(name="src.tasks.embedding.batch_process_embeddings")
def batch_process_embeddings(paper_ids: List[str]) -> dict:
    """
    Process embeddings for multiple papers.

    Args:
        paper_ids: List of paper IDs

    Returns:
        Batch processing results
    """
    results = []

    for paper_id in paper_ids:
        result = process_paper_embeddings.delay(paper_id)
        results.append({
            "paper_id": paper_id,
            "task_id": result.id
        })

    return {
        "papers_queued": len(paper_ids),
        "tasks": results
    }


@celery_app.task(name="src.tasks.embedding.reindex_all")
def reindex_all() -> dict:
    """
    Reindex all papers in the vector store.

    WARNING: This clears the existing index!
    """
    result = asyncio.run(_async_reindex_all())
    return result


async def _async_reindex_all() -> dict:
    """Async implementation of full reindexing."""
    from src.core.database import AsyncSessionLocal
    from src.services.paper_service import PaperService
    from src.services.storage.vector_store import VectorStore

    try:
        # Clear existing vector store
        vector_store = VectorStore()
        vector_store.clear()
        logger.info("Cleared existing vector store")

        # Fetch all papers from database
        async with AsyncSessionLocal() as db:
            paper_service = PaperService(db)
            papers = await paper_service.list_all(limit=10000)

        if not papers:
            return {
                "status": "completed",
                "message": "No papers to reindex",
                "papers_processed": 0
            }

        # Queue embedding generation for all papers
        queued = 0
        for paper in papers:
            process_paper_embeddings.delay(str(paper.id))
            queued += 1

        logger.info(f"Queued {queued} papers for reindexing")

        return {
            "status": "completed",
            "message": f"Queued {queued} papers for reindexing",
            "papers_processed": queued
        }

    except Exception as e:
        logger.error(f"Error during reindex: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
