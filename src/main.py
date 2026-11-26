"""FastAPI application entry point for NLPulse."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from src.api.routes.batch import router as batch_router
from src.api.routes.developers import router as developers_router
from src.api.routes.health import router as health_router
from src.api.routes.keys import router as keys_router
from src.api.routes.models import router as models_router
from src.api.routes.nlp import router as nlp_router
from src.api.routes.protected import router as protected_router
from src.api.routes.usage import router as usage_router
from src.core.config import get_settings
from src.core.device import DeviceManager
from src.core.exceptions import NLPulseException, RateLimitExceededError
from src.core.logging import get_logger, setup_logging
from src.core.models.base import ModelMetadata
from src.core.models.nlp import (
    EmbeddingModel,
    NERModel,
    SentimentModel,
    SummarizationModel,
    ZeroShotClassificationModel,
)
from src.db.session import engine
from src.services.model_registry import ModelRegistry

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


def _register_default_models(registry: ModelRegistry) -> None:
    """Register standard M4 NLP model definitions into the registry for lazy loading."""
    # 1. Sentiment Analysis
    sentiment_meta = ModelMetadata(
        model_id=settings.MODEL_SENTIMENT_ID,
        name="DistilBERT Sentiment Analysis",
        task="sentiment",
        version="1.0.0",
        description="DistilBERT fine-tuned on SST-2 for sentiment classification",
        vram_estimate_mb=270.0,
        ram_estimate_mb=350.0,
        max_concurrency=settings.MODEL_DEFAULT_CONCURRENCY,
        timeout_seconds=settings.MODEL_INFERENCE_TIMEOUT_SECONDS,
    )
    if not registry.is_registered(settings.MODEL_SENTIMENT_ID):
        registry.register(sentiment_meta, lambda: SentimentModel(sentiment_meta))

    # 2. Named Entity Recognition
    ner_meta = ModelMetadata(
        model_id=settings.MODEL_NER_ID,
        name="BERT Named Entity Recognition",
        task="ner",
        version="1.0.0",
        description="BERT fine-tuned on CoNLL-2003 for entity span extraction",
        vram_estimate_mb=420.0,
        ram_estimate_mb=500.0,
        max_concurrency=settings.MODEL_DEFAULT_CONCURRENCY,
        timeout_seconds=settings.MODEL_INFERENCE_TIMEOUT_SECONDS,
    )
    if not registry.is_registered(settings.MODEL_NER_ID):
        registry.register(ner_meta, lambda: NERModel(ner_meta))

    # 3. Zero-Shot Classification
    zero_shot_meta = ModelMetadata(
        model_id=settings.MODEL_ZERO_SHOT_ID,
        name="DistilBART Zero-Shot Classification",
        task="zero-shot",
        version="1.0.0",
        description="DistilBART fine-tuned on MNLI for zero-shot text classification",
        vram_estimate_mb=550.0,
        ram_estimate_mb=700.0,
        max_concurrency=settings.MODEL_DEFAULT_CONCURRENCY,
        timeout_seconds=settings.MODEL_INFERENCE_TIMEOUT_SECONDS,
    )
    if not registry.is_registered(settings.MODEL_ZERO_SHOT_ID):
        registry.register(zero_shot_meta, lambda: ZeroShotClassificationModel(zero_shot_meta))

    # 4. Sentence Embeddings & Keyword Extraction (M5)
    embedding_meta = ModelMetadata(
        model_id=settings.MODEL_EMBEDDING_ID,
        name="MiniLM Sentence Embeddings",
        task="embedding",
        version="1.0.0",
        description="All-MiniLM-L6-v2 sentence embedding model for dense representation and semantic search",
        vram_estimate_mb=90.0,
        ram_estimate_mb=120.0,
        max_concurrency=settings.MODEL_DEFAULT_CONCURRENCY,
        timeout_seconds=settings.MODEL_INFERENCE_TIMEOUT_SECONDS,
    )
    if not registry.is_registered(settings.MODEL_EMBEDDING_ID):
        registry.register(embedding_meta, lambda: EmbeddingModel(embedding_meta))

    # 5. Abstractive Text Summarization (M5)
    summarization_meta = ModelMetadata(
        model_id=settings.MODEL_SUMMARIZATION_ID,
        name="DistilBART CNN Summarization",
        task="summarization",
        version="1.0.0",
        description="DistilBART fine-tuned on CNN/DailyMail for abstractive document summarization",
        vram_estimate_mb=600.0,
        ram_estimate_mb=1200.0,
        max_concurrency=settings.MODEL_DEFAULT_CONCURRENCY,
        timeout_seconds=settings.MODEL_INFERENCE_TIMEOUT_SECONDS,
    )
    if not registry.is_registered(settings.MODEL_SUMMARIZATION_ID):
        registry.register(summarization_meta, lambda: SummarizationModel(summarization_meta))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager for startup and shutdown events."""
    setup_logging(log_level=settings.LOG_LEVEL, log_format=settings.LOG_FORMAT)

    # Initialize compute device and model registry in application state
    device_mgr = DeviceManager(preference=settings.MODEL_DEVICE_PREFERENCE)
    registry = ModelRegistry(device_mgr=device_mgr, max_cached_models=settings.MODEL_MAX_CACHED)

    # Register core NLP models (lazy loaded on demand, zero weight download on startup)
    _register_default_models(registry)

    app.state.device_manager = device_mgr
    app.state.model_registry = registry

    hw = device_mgr.get_hardware_info()
    logger.info(
        "NLPulse service starting up",
        extra={
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "env": settings.APP_ENV,
            "device": hw.device_type,
            "cuda_available": hw.cuda_available,
            "device_name": hw.device_name,
        },
    )
    yield
    logger.info("NLPulse service shutting down")
    if hasattr(app.state, "model_registry") and app.state.model_registry:
        await app.state.model_registry.unload_all()
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
    app.include_router(models_router, prefix=settings.API_V1_PREFIX)
    app.include_router(batch_router, prefix=settings.API_V1_PREFIX)
    app.include_router(nlp_router, prefix=settings.API_V1_PREFIX)
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
