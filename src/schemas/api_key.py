"""Schemas for API key issuance, listing, and revocation."""

from datetime import datetime
from typing import List, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field


class APIKeyCreate(BaseModel):
    """Payload for creating a new API key."""

    name: str = Field(..., min_length=1, max_length=100, description="Friendly label for the API key")
    scopes: List[str] = Field(
        default=["analyze:read"],
        description="Permission scopes granted to this key (e.g. sentiment:read, ner:read, admin)",
    )
    expires_in_days: Optional[int] = Field(
        default=None,
        ge=1,
        le=365,
        description="Optional expiration window in days (null for non-expiring keys)",
    )


class APIKeyResponse(BaseModel):
    """Public representation of an API key (never exposes raw key or secret hash)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    developer_id: uuid.UUID
    key_prefix: str
    name: str
    scopes: List[str]
    is_active: bool
    created_at: datetime
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None


class APIKeyCreatedResponse(APIKeyResponse):
    """Returned ONLY ONCE at key creation time containing the plaintext secret."""

    api_key: str = Field(
        ...,
        description="Plaintext API key. Store this securely; it will not be displayed again.",
    )
