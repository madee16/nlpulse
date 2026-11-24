"""Developer registration and management routes."""

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_developer, get_db
from src.models.developer import Developer
from src.schemas.developer import (
    DeveloperCreate,
    DeveloperRegistrationResponse,
    DeveloperResponse,
)
from src.services.api_key_service import APIKeyService
from src.services.developer_service import DeveloperService

router = APIRouter(prefix="/developers", tags=["Developers"])


@router.post(
    "",
    response_model=DeveloperRegistrationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new developer",
    description=(
        "Registers a developer account on the free tier and automatically issues the initial "
        "bootstrap API key. The plaintext API key is returned strictly once in this response."
    ),
)
async def register_developer(
    payload: DeveloperCreate,
    db: AsyncSession = Depends(get_db),
) -> DeveloperRegistrationResponse:
    """Register a new developer tenant and issue their initial API key."""
    dev_service = DeveloperService(db)
    # Security: Public registration always forces the 'free' tier and 1,000 quota
    developer = await dev_service.create_developer(
        email=payload.email,
        name=payload.name,
        tier="free",
    )

    # Automatically generate the initial bootstrap API key with standard scopes
    key_service = APIKeyService(db)
    api_key, raw_key = await key_service.create_key(
        developer_id=developer.id,
        name="Default API Key",
        scopes=[
            "analyze:read",
            "sentiment:read",
            "ner:read",
            "classification:read",
            "keywords:read",
            "summarization:read",
            "extraction:read",
            "generate:read",
            "keys:read",
            "keys:write",
            "usage:read",
        ],
    )

    data = {
        "id": developer.id,
        "email": developer.email,
        "name": developer.name,
        "tier": developer.tier,
        "monthly_request_quota": developer.monthly_request_quota,
        "is_active": developer.is_active,
        "created_at": developer.created_at,
        "updated_at": developer.updated_at,
        "api_key": raw_key,
        "key_prefix": api_key.key_prefix,
        "key_id": api_key.id,
    }
    return DeveloperRegistrationResponse.model_validate(data)


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
