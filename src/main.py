"""FastAPI application entry point for NLPulse."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from src.api.routes.developers import router as developers_router
from src.api.routes.health import router as health_router
from src.api.routes.keys import router as keys_router
from src.api.routes.protected import router as protected_router
from src.api.routes.usage import router as usage_router
from src.core.config import get_settings
from src.core.exceptions import NLPulseException, RateLimitExceededError
from src.core.logging import get_logger, setup_logging
from src.db.session import engine

settings = get_settings()
logger = get_logger("nlpulse.main")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Enforce standard security headers across all HTTP responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


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
    await engine.dispose()


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

    # Middleware
    app.add_middleware(SecurityHeadersMiddleware)

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

    # Global Exception Handlers
    @app.exception_handler(NLPulseException)
    async def nlp_exception_handler(request: Request, exc: NLPulseException) -> JSONResponse:
        headers = {}
        if isinstance(exc, RateLimitExceededError):
            headers["Retry-After"] = str(exc.retry_after)

        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "status_code": exc.status_code},
            headers=headers,
        )

    # Register API routers
    app.include_router(health_router, prefix=settings.API_V1_PREFIX)
    app.include_router(health_router)  # root convenience
    app.include_router(developers_router, prefix=settings.API_V1_PREFIX)
    app.include_router(keys_router, prefix=settings.API_V1_PREFIX)
    app.include_router(usage_router, prefix=settings.API_V1_PREFIX)
    app.include_router(protected_router, prefix=settings.API_V1_PREFIX)

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
