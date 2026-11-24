"""Schemas for developer registration and profile management."""

from datetime import datetime
from typing import Optional
import uuid

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class DeveloperCreate(BaseModel):
    """Payload for registering a new developer."""

    email: EmailStr = Field(..., description="Developer contact and login email")
    name: str = Field(..., min_length=2, max_length=100, description="Developer or team name")
    tier: Optional[str] = Field("free", description="Subscription tier (free, pro, enterprise)")


class DeveloperResponse(BaseModel):
    """Developer profile representation."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    name: str
    tier: str
    monthly_request_quota: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
