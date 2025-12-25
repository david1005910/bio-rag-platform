"""Library API Endpoints - User's saved papers"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.core.security import get_current_user_id

router = APIRouter()


# ============== Schemas ==============

class SavePaperRequest(BaseModel):
    """Save paper request"""
    pmid: str
    tags: List[str] = []
    notes: Optional[str] = None


class SavedPaper(BaseModel):
    """Saved paper"""
    id: str
    pmid: str
    title: str
    abstract: str
    tags: List[str] = []
    notes: Optional[str] = None
    saved_at: str


class SavedPaperListResponse(BaseModel):
    """Saved paper list response"""
    total: int
    papers: List[SavedPaper]


class TagListResponse(BaseModel):
    """Tag list response"""
    tags: List[str]


class UpdatePaperRequest(BaseModel):
    """Update saved paper request"""
    tags: Optional[List[str]] = None
    notes: Optional[str] = None


# ============== Endpoints ==============

@router.post("/papers", response_model=SavedPaper)
async def save_paper(
    request: SavePaperRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Save a paper to user's library

    - Requires authentication
    - Can add tags and notes
    """
    import uuid
    from datetime import datetime

    # TODO: Save to database

    return SavedPaper(
        id=str(uuid.uuid4()),
        pmid=request.pmid,
        title="Sample Paper Title",
        abstract="Paper abstract...",
        tags=request.tags,
        notes=request.notes,
        saved_at=datetime.utcnow().isoformat()
    )


@router.get("/papers", response_model=SavedPaperListResponse)
async def get_saved_papers(
    user_id: str = Depends(get_current_user_id),
    tag: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0)
):
    """
    Get user's saved papers

    - Requires authentication
    - Supports filtering by tag
    - Supports pagination
    """
    # TODO: Fetch from database

    return SavedPaperListResponse(
        total=0,
        papers=[]
    )


@router.get("/papers/{paper_id}", response_model=SavedPaper)
async def get_saved_paper(
    paper_id: str,
    user_id: str = Depends(get_current_user_id)
):
    """
    Get a specific saved paper

    - Requires authentication
    """
    raise HTTPException(
        status_code=404,
        detail=f"Saved paper {paper_id} not found"
    )


@router.put("/papers/{paper_id}", response_model=SavedPaper)
async def update_saved_paper(
    paper_id: str,
    request: UpdatePaperRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Update a saved paper (tags, notes)

    - Requires authentication
    """
    raise HTTPException(
        status_code=404,
        detail=f"Saved paper {paper_id} not found"
    )


@router.delete("/papers/{paper_id}")
async def delete_saved_paper(
    paper_id: str,
    user_id: str = Depends(get_current_user_id)
):
    """
    Remove a paper from library

    - Requires authentication
    """
    # TODO: Delete from database

    return {"message": f"Paper {paper_id} removed from library"}


@router.get("/tags", response_model=TagListResponse)
async def get_tags(
    user_id: str = Depends(get_current_user_id)
):
    """
    Get all tags used by user

    - Requires authentication
    """
    # TODO: Fetch from database

    return TagListResponse(tags=[])
