"""API key management endpoints for authenticated developers."""

from typing import List
import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_api_key, get_current_developer, get_db, require_scopes
from src.core.exceptions import PermissionDeniedError
from src.models.api_key import APIKey
from src.models.developer import Developer
from src.schemas.api_key import (
    APIKeyCreate,
    APIKeyCreatedResponse,
    APIKeyResponse,
)
from src.services.api_key_service import APIKeyService

router = APIRouter(prefix="/keys", tags=["API Keys"])


@router.post(
    "",
    response_model=APIKeyCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new API key",
    description=(
        "Generates a new API key for the authenticated developer. Raw key returned ONLY once. "
        "Keys can only be granted a subset of the scopes held by the authorizing API key."
    ),
    dependencies=[Depends(require_scopes("keys:write", "admin"))],
)
async def create_key(
    payload: APIKeyCreate,
    current_key: APIKey = Depends(get_current_api_key),
    developer: Developer = Depends(get_current_developer),
    db: AsyncSession = Depends(get_db),
) -> APIKeyCreatedResponse:
    """Create a new API key for the current authenticated developer preventing privilege escalation."""
    requested_scopes = set(payload.scopes)
    authorizing_scopes = set(current_key.scopes or [])

    # Privilege Escalation Guard:
    # A standard key without 'admin' cannot grant 'admin' or any scope it does not possess
    if "admin" not in authorizing_scopes and "*" not in authorizing_scopes:
        if "admin" in requested_scopes or "*" in requested_scopes:
            raise PermissionDeniedError("Cannot grant unassigned or privileged scope: 'admin'.")

        unauthorized_scopes = requested_scopes - authorizing_scopes
        if unauthorized_scopes:
            raise PermissionDeniedError(
                f"Cannot grant unassigned or privileged scope(s) {list(unauthorized_scopes)}. "
                f"Authorizing key only holds: {list(authorizing_scopes)}"
            )

    service = APIKeyService(db)
    api_key, raw_key = await service.create_key(
        developer_id=developer.id,
        name=payload.name,
        scopes=list(requested_scopes),
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


@router.get(
    "",
    response_model=List[APIKeyResponse],
    summary="List API keys",
    description="Lists all API keys belonging to the authenticated developer (secrets are never returned).",
    dependencies=[Depends(require_scopes("keys:read", "keys:write", "admin"))],
)
async def list_keys(
    developer: Developer = Depends(get_current_developer),
    db: AsyncSession = Depends(get_db),
) -> List[APIKeyResponse]:
    """List all API keys belonging to the current developer."""
    service = APIKeyService(db)
    keys = await service.list_keys(developer.id)
    return [APIKeyResponse.model_validate(k) for k in keys]


@router.delete(
    "/{key_id}",
    response_model=APIKeyResponse,
    summary="Revoke an API key",
    description="Deactivates an API key. Any future requests with this key will fail immediately.",
    dependencies=[Depends(require_scopes("keys:write", "admin"))],
)
async def revoke_key(
    key_id: uuid.UUID,
    developer: Developer = Depends(get_current_developer),
    db: AsyncSession = Depends(get_db),
) -> APIKeyResponse:
    """Revoke an API key owned by the authenticated developer."""
    service = APIKeyService(db)
    revoked_key = await service.revoke_key(
        developer_id=developer.id,
        key_id=key_id,
    )
    return APIKeyResponse.model_validate(revoked_key)
