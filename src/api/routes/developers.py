"""Developer registration and management routes."""

from typing import List
import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_developer, get_db
from src.models.developer import Developer
from src.schemas.api_key import APIKeyCreate, APIKeyCreatedResponse
from src.schemas.developer import DeveloperCreate, DeveloperResponse
from src.services.api_key_service import APIKeyService
from src.services.developer_service import DeveloperService

router = APIRouter(prefix="/developers", tags=["Developers"])


@router.post(
    "",
    response_model=DeveloperResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new developer",
    description="Creates a developer organization account for API access.",
)
async def register_developer(
    payload: DeveloperCreate,
    db: AsyncSession = Depends(get_db),
) -> DeveloperResponse:
    """Register a new developer tenant."""
    service = DeveloperService(db)
    developer = await service.create_developer(
        email=payload.email,
        name=payload.name,
        tier=payload.tier or "free",
    )
    return DeveloperResponse.model_validate(developer)


@router.get(
    "/me",
    response_model=DeveloperResponse,
    summary="Get authenticated developer profile",
    description="Returns the profile and quota of the developer owning the provided API key.",
)
async def get_my_profile(
    developer: Developer = Depends(get_current_developer),
) -> DeveloperResponse:
    """Get the authenticated developer's profile."""
    return DeveloperResponse.model_validate(developer)


@router.post(
    "/{developer_id}/keys",
    response_model=APIKeyCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Issue an initial API key for a developer",
    description="Generates an initial API key. The plaintext key is returned ONLY once in this response.",
)
async def issue_developer_key(
    developer_id: uuid.UUID,
    payload: APIKeyCreate,
    db: AsyncSession = Depends(get_db),
) -> APIKeyCreatedResponse:
    """Bootstrap or issue a key for a specific developer ID."""
    dev_service = DeveloperService(db)
    # Ensure developer exists
    await dev_service.get_required_developer(developer_id)

    key_service = APIKeyService(db)
    api_key, raw_key = await key_service.create_key(
        developer_id=developer_id,
        name=payload.name,
        scopes=payload.scopes,
        expires_in_days=payload.expires_in_days,
    )

    data = {
        "id": api_key.id,
        "developer_id": api_key.developer_id,
        "key_prefix": api_key.key_prefix,
        "name": api_key.name,
        "scopes": api_key.scopes,
        "is_active": api_key.is_active,
        "created_at": api_key.created_at,
        "expires_at": api_key.expires_at,
        "revoked_at": api_key.revoked_at,
        "last_used_at": api_key.last_used_at,
        "api_key": raw_key,
    }
    return APIKeyCreatedResponse.model_validate(data)
