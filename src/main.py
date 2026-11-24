"""FastAPI application entry point for NLPulse."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.health import router as health_router
from src.core.config import get_settings
from src.core.logging import get_logger, setup_logging

settings = get_settings()
logger = get_logger("nlpulse.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager for startup and shutdown events."""
    setup_logging(log_level=settings.LOG_LEVEL, log_format=settings.LOG_FORMAT)
    logger.info(
        "NLPulse service starting up",
        extra={
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "env": settings.APP_ENV,
        },
    )
    yield
    logger.info("NLPulse service shutting down")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "NLPulse — Intelligent Text Analysis API (NLP-as-a-Service). "
            "Provides high-performance NLP features including sentiment analysis, "
            "NER, zero-shot classification, keyword extraction, and summarization."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS
    cors_origins = (
        settings.CORS_ORIGINS
        if isinstance(settings.CORS_ORIGINS, list)
        else [settings.CORS_ORIGINS]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routers
    app.include_router(health_router, prefix=settings.API_V1_PREFIX)
    # Also mount health at root for container orchestration convenience
    app.include_router(health_router)

    @app.get("/", tags=["Root"], summary="API Root")
    async def root():
        """Root endpoint providing service metadata and discovery links."""
        return {
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.APP_ENV,
            "status": "operational",
            "docs_url": "/docs",
            "health_url": f"{settings.API_V1_PREFIX}/health",
        }

    return app


app = create_application()
