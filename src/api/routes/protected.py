"""Protected sample routes verifying auth, scope permissions, and rate limits."""

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import (
    check_quota,
    check_rate_limit,
    get_current_api_key,
    get_current_developer,
    get_db,
    require_scopes,
)
from src.models.api_key import APIKey
from src.models.developer import Developer
from src.services.usage_service import UsageService

router = APIRouter(prefix="/protected", tags=["Protected Foundations"])


@router.get(
    "/sentiment",
    dependencies=[
        Depends(check_rate_limit),
        Depends(check_quota),
        Depends(require_scopes("sentiment:read", "analyze:read")),
    ],
    summary="Protected sentiment test endpoint",
    description="Demonstrates scope checking, rate limiting, and quota enforcement.",
)
async def protected_sentiment_demo(
    request: Request,
    api_key: APIKey = Depends(get_current_api_key),
    developer: Developer = Depends(get_current_developer),
    db: AsyncSession = Depends(get_db),
):
    """Test endpoint requiring sentiment:read or analyze:read scope."""
    usage_service = UsageService(db)
    await usage_service.record_usage(
        developer_id=developer.id,
        api_key_id=api_key.id,
        endpoint=str(request.url.path),
        method=request.method,
        status_code=200,
        latency_ms=1.5,
    )
    return {
        "status": "authorized",
        "message": "Access granted to sentiment endpoint",
        "developer": developer.name,
        "key_prefix": api_key.key_prefix,
    }


@router.get(
    "/admin-only",
    dependencies=[
        Depends(check_rate_limit),
        Depends(check_quota),
        Depends(require_scopes("admin")),
    ],
    summary="Protected admin-only test endpoint",
    description="Demonstrates strict scope checking requiring admin privileges.",
)
async def protected_admin_demo(
    developer: Developer = Depends(get_current_developer),
):
    """Test endpoint requiring admin scope."""
    return {
        "status": "authorized",
        "message": "Access granted to admin endpoint",
        "developer": developer.name,
    }
