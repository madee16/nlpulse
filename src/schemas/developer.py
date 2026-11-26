"""Schemas for developer registration and profile management."""

from datetime import datetime
from typing import Optional
import uuid

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class DeveloperCreate(BaseModel):
    """Payload for registering a new developer."""

    email: EmailStr = Field(..., description="Developer contact and login email")
    name: str = Field(..., min_length=2, max_length=100, description="Developer or team name")
    # Note: Subscription tier is strictly controlled server-side. Public registration always assigns 'free'.

    @field_validator("name")
    @classmethod
    def validate_name_non_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if len(cleaned) < 2:
            raise ValueError("Developer name must contain at least 2 non-whitespace characters.")
        return cleaned


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


class DeveloperRegistrationResponse(DeveloperResponse):
    """Returned upon successful registration, including the initial bootstrap API key."""

    api_key: str = Field(
        ...,
        description="Initial plaintext API key. Save this securely; it will NOT be displayed again.",
    )
    key_prefix: str = Field(..., description="Public prefix of the generated key")
    key_id: uuid.UUID = Field(..., description="UUID identifier of the generated key")
