"""Paper Database Service - CRUD operations for papers"""

import logging
from typing import Optional, List
from uuid import UUID
from datetime import date

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.paper import Paper, Chunk

logger = logging.getLogger(__name__)


class PaperService:
    """Service for paper database operations"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, paper_id: UUID) -> Optional[Paper]:
        """Get paper by ID"""
        result = await self.db.execute(
            select(Paper).where(Paper.id == paper_id)
        )
        return result.scalar_one_or_none()

    async def get_by_pmid(self, pmid: str) -> Optional[Paper]:
        """Get paper by PMID"""
        result = await self.db.execute(
            select(Paper).where(Paper.pmid == pmid)
        )
        return result.scalar_one_or_none()

    async def create(
        self,
        pmid: str,
        title: str,
        abstract: Optional[str] = None,
        authors: Optional[List[str]] = None,
        journal: Optional[str] = None,
        publication_date: Optional[date] = None,
        doi: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        mesh_terms: Optional[List[str]] = None,
    ) -> Paper:
        """Create a new paper"""
        paper = Paper(
            pmid=pmid,
            title=title,
            abstract=abstract,
            authors=authors or [],
            journal=journal,
            publication_date=publication_date,
            doi=doi,
            keywords=keywords or [],
            mesh_terms=mesh_terms or [],
        )
        self.db.add(paper)
        await self.db.flush()
        await self.db.refresh(paper)
        logger.info(f"Created paper: {pmid}")
        return paper

    async def get_or_create(
        self,
        pmid: str,
        title: str,
        abstract: Optional[str] = None,
        authors: Optional[List[str]] = None,
        journal: Optional[str] = None,
        publication_date: Optional[date] = None,
        doi: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        mesh_terms: Optional[List[str]] = None,
    ) -> tuple[Paper, bool]:
        """Get existing paper or create new one. Returns (paper, created)"""
        existing = await self.get_by_pmid(pmid)
        if existing:
            return existing, False

        paper = await self.create(
            pmid=pmid,
            title=title,
            abstract=abstract,
            authors=authors,
            journal=journal,
            publication_date=publication_date,
            doi=doi,
            keywords=keywords,
            mesh_terms=mesh_terms,
        )
        return paper, True

    async def update(self, paper: Paper, **kwargs) -> Paper:
        """Update paper fields"""
        for key, value in kwargs.items():
            if hasattr(paper, key):
                setattr(paper, key, value)
        await self.db.flush()
        await self.db.refresh(paper)
        return paper

    async def delete(self, paper_id: UUID) -> bool:
        """Delete paper by ID"""
        result = await self.db.execute(
            delete(Paper).where(Paper.id == paper_id)
        )
        return result.rowcount > 0

    async def list_all(self, limit: int = 100, offset: int = 0) -> List[Paper]:
        """List all papers with pagination"""
        result = await self.db.execute(
            select(Paper).order_by(Paper.created_at.desc()).limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    async def count(self) -> int:
        """Count total papers"""
        from sqlalchemy import func
        result = await self.db.execute(select(func.count(Paper.id)))
        return result.scalar() or 0

    async def add_chunk(
        self,
        paper_id: UUID,
        text: str,
        section: str,
        chunk_index: int,
        token_count: Optional[int] = None,
        embedding_id: Optional[str] = None,
    ) -> Chunk:
        """Add a chunk to a paper"""
        chunk = Chunk(
            paper_id=paper_id,
            text=text,
            section=section,
            chunk_index=chunk_index,
            token_count=token_count,
            embedding_id=embedding_id,
        )
        self.db.add(chunk)
        await self.db.flush()
        await self.db.refresh(chunk)
        return chunk

    async def get_chunks(self, paper_id: UUID) -> List[Chunk]:
        """Get all chunks for a paper"""
        result = await self.db.execute(
            select(Chunk)
            .where(Chunk.paper_id == paper_id)
            .order_by(Chunk.chunk_index)
        )
        return list(result.scalars().all())

    async def delete_chunks(self, paper_id: UUID) -> int:
        """Delete all chunks for a paper"""
        result = await self.db.execute(
            delete(Chunk).where(Chunk.paper_id == paper_id)
        )
        return result.rowcount
