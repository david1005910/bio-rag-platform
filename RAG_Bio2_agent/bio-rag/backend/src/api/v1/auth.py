"""Authentication API Endpoints"""

from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.core.security import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user_id,
)
from src.core.config import settings

router = APIRouter()


# ============== Schemas ==============

class UserRegisterRequest(BaseModel):
    """User registration request"""
    email: EmailStr
    password: str
    name: str
    research_field: Optional[str] = None


class UserLoginRequest(BaseModel):
    """User login request"""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    """User response"""
    id: str
    email: str
    name: str
    research_field: Optional[str] = None


class MessageResponse(BaseModel):
    """Generic message response"""
    message: str


# ============== Endpoints ==============

@router.post("/register", response_model=TokenResponse)
async def register(
    request: UserRegisterRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user

    - Creates user account
    - Returns access token
    """
    # TODO: Check if user exists
    # TODO: Create user in database

    # For now, create token directly
    access_token = create_access_token(
        data={"sub": request.email},
        expires_delta=timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    )

    return TokenResponse(
        access_token=access_token,
        expires_in=settings.JWT_EXPIRE_MINUTES * 60
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    request: UserLoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Authenticate user and return access token

    - Verifies email and password
    - Returns JWT access token
    """
    # TODO: Verify user credentials from database

    # For now, create token directly
    access_token = create_access_token(
        data={"sub": request.email},
        expires_delta=timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    )

    return TokenResponse(
        access_token=access_token,
        expires_in=settings.JWT_EXPIRE_MINUTES * 60
    )


@router.post("/logout", response_model=MessageResponse)
async def logout(
    current_user_id: str = Depends(get_current_user_id)
):
    """
    Logout user

    - Invalidates the current token (client should discard it)
    """
    # JWT tokens are stateless, so we just return success
    # For proper logout, implement token blacklist with Redis
    return MessageResponse(message="Successfully logged out")


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Get current user profile

    - Requires authentication
    """
    # TODO: Fetch user from database
    return UserResponse(
        id=current_user_id,
        email=current_user_id,
        name="User",
        research_field=None
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    current_user_id: str = Depends(get_current_user_id)
):
    """
    Refresh access token

    - Requires valid current token
    - Returns new access token
    """
    access_token = create_access_token(
        data={"sub": current_user_id},
        expires_delta=timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    )

    return TokenResponse(
        access_token=access_token,
        expires_in=settings.JWT_EXPIRE_MINUTES * 60
    )
