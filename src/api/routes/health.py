"""Health check endpoint for service monitoring."""

from datetime import datetime, timezone
from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.core.config import get_settings

router = APIRouter(tags=["Health"])
settings = get_settings()


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(default="healthy", description="Service health status")
    service: str = Field(default=settings.APP_NAME, description="Service name")
    version: str = Field(default=settings.APP_VERSION, description="Application version")
    environment: str = Field(default=settings.APP_ENV, description="Active environment")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp of the health check",
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the current operational status and version of the NLPulse API.",
)
async def health_check() -> HealthResponse:
    """Return health status of the application."""
    return HealthResponse(
        status="healthy",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
    )
