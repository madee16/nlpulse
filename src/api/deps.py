"""FastAPI dependencies for database access, authentication, scopes, and quota enforcement."""

import asyncio
from typing import AsyncGenerator, List, Optional

from fastapi import Depends, Header, HTTPException, Request, Security
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import (
    AuthenticationError,
    PermissionDeniedError,
    QuotaExceededError,
    RateLimitExceededError,
)
from src.core.rate_limiter import rate_limiter
from src.db.session import get_db
from src.models.api_key import APIKey
from src.models.developer import Developer
from src.services.api_key_service import APIKeyService
from src.services.developer_service import DeveloperService
from src.services.model_registry import ModelRegistry, model_registry
from src.services.ollama_client import OllamaClient
from src.services.usage_service import UsageService

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_api_key(
    raw_key: Optional[str] = Security(api_key_header),
    db: AsyncSession = Depends(get_db),
) -> APIKey:
    """Extract and validate the API key from the X-API-Key request header."""
    if not raw_key:
        raise AuthenticationError("API key required. Provide 'X-API-Key' in request headers.")

    key_service = APIKeyService(db)
    return await key_service.authenticate_raw_key(raw_key)


async def get_current_developer(
    api_key: APIKey = Depends(get_current_api_key),
    db: AsyncSession = Depends(get_db),
) -> Developer:
    """Retrieve the active tenant Developer linked to the authenticated API key."""
    dev_service = DeveloperService(db)
    developer = await dev_service.get_by_id(api_key.developer_id)
    if not developer or not developer.is_active:
        raise AuthenticationError("Developer account is inactive or not found.")
    return developer


def require_scopes(*required_scopes: str):
    """Dependency factory ensuring the authenticated API key has sufficient scopes."""

    async def scope_checker(api_key: APIKey = Depends(get_current_api_key)) -> None:
        key_scopes = set(api_key.scopes or [])
        # 'admin' or wildcard '*' satisfies all requirements
        if "admin" in key_scopes or "*" in key_scopes:
            return

        # Satisfied if key has at least one of the authorized scopes
        if not any(req in key_scopes for req in required_scopes):
            raise PermissionDeniedError(
                f"Forbidden: API key lacks required scope(s) {list(required_scopes)}. Granted scopes: {list(key_scopes)}"
            )

    return scope_checker


async def check_quota(
    developer: Developer = Depends(get_current_developer),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Evaluate whether the developer has exceeded their monthly request quota."""
    usage_service = UsageService(db)
    await usage_service.assert_within_quota(developer)


async def reserve_single_quota(
    developer: Developer = Depends(get_current_developer),
    db: AsyncSession = Depends(get_db),
) -> AsyncGenerator[None, None]:
    """Reserve 1 quota unit across single-item inference to eliminate concurrent TOCTOU races."""
    usage_service = UsageService(db)
    await usage_service.reserve_quota(developer, units=1)
    session_info = getattr(db, "info", None)
    reservations = session_info.setdefault("_quota_reservations", {}) if isinstance(session_info, dict) else None
    if isinstance(reservations, dict):
        reservations[developer.id] = reservations.get(developer.id, 0) + 1
    try:
        yield
    finally:
        if isinstance(reservations, dict):
            remaining = reservations.get(developer.id, 0)
            if remaining > 0:
                release_units = 1
                reservations[developer.id] = remaining - release_units
                if reservations[developer.id] <= 0:
                    reservations.pop(developer.id, None)
                await asyncio.shield(usage_service.release_reservation(developer.id, units=release_units))
        else:
            await asyncio.shield(usage_service.release_reservation(developer.id, units=1))


async def check_rate_limit(
    api_key: APIKey = Depends(get_current_api_key),
) -> None:
    """Process-local rate limiting based on API key prefix/hash."""
    is_allowed, retry_after = await rate_limiter.check(api_key.key_hash)
    if not is_allowed:
        raise RateLimitExceededError(retry_after=retry_after)


def get_model_registry(request: Request) -> ModelRegistry:
    """Provide the application-scoped ModelRegistry instance from state or fallback to default."""
    if hasattr(request.app.state, "model_registry") and request.app.state.model_registry:
        return request.app.state.model_registry
    return model_registry


def get_ollama_client(request: Request) -> OllamaClient:
    """Provide the application-scoped OllamaClient instance from state or fallback to default."""
    if hasattr(request.app.state, "ollama_client") and request.app.state.ollama_client:
        return request.app.state.ollama_client
    return OllamaClient()

