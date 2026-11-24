"""Usage statistics and quota summary endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_developer, get_db
from src.models.developer import Developer
from src.schemas.usage import UsageSummaryResponse
from src.services.usage_service import UsageService

router = APIRouter(prefix="/usage", tags=["Usage & Quotas"])


@router.get(
    "/summary",
    response_model=UsageSummaryResponse,
    summary="Get monthly quota usage summary",
    description="Returns current calendar month request consumption, quota limit, and remaining balance.",
)
async def get_usage_summary(
    developer: Developer = Depends(get_current_developer),
    db: AsyncSession = Depends(get_db),
) -> UsageSummaryResponse:
    """Get quota and usage summary for the current developer."""
    service = UsageService(db)
    return await service.get_usage_summary(developer)
