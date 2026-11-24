"""FastAPI dependencies for database access, authentication, scopes, and quota enforcement."""

from typing import AsyncGenerator, List, Optional

from fastapi import Depends, Header, HTTPException, Security
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


async def check_rate_limit(
    api_key: APIKey = Depends(get_current_api_key),
) -> None:
    """Process-local rate limiting based on API key prefix/hash."""
    is_allowed, retry_after = await rate_limiter.check(api_key.key_hash)
    if not is_allowed:
        raise RateLimitExceededError(retry_after=retry_after)
