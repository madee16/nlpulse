"""API endpoints for M7 batch NLP processing across all six capabilities."""

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import (
    check_quota,
    check_rate_limit,
    get_current_api_key,
    get_current_developer,
    get_model_registry,
    get_ollama_client,
    require_scopes,
)
from src.db.session import get_db
from src.models.api_key import APIKey
from src.models.developer import Developer
from src.schemas.batch import (
    BatchExtractionRequest,
    BatchKeywordRequest,
    BatchNERRequest,
    BatchResponse,
    BatchSentimentRequest,
    BatchSummarizationRequest,
    BatchZeroShotRequest,
)
from src.schemas.extraction import ExtractionResponse
from src.schemas.nlp import (
    KeywordResponse,
    NERResponse,
    SentimentResponse,
    SummarizationResponse,
    ZeroShotResponse,
)
from src.services.batch_service import BatchService
from src.services.model_registry import ModelRegistry
from src.services.ollama_client import OllamaClient

router = APIRouter(prefix="/nlp/batch", tags=["Batch NLP Processing (M7)"])


@router.post(
    "/sentiment",
    response_model=BatchResponse[SentimentResponse],
    status_code=status.HTTP_200_OK,
    summary="Batch sentiment analysis",
    dependencies=[
        Depends(require_scopes("sentiment:read", "nlp:read", "analyze:read", "admin")),
        Depends(check_rate_limit),
        Depends(check_quota),
    ],
)
async def batch_analyze_sentiment(
    payload: BatchSentimentRequest,
    current_key: APIKey = Depends(get_current_api_key),
    developer: Developer = Depends(get_current_developer),
    db: AsyncSession = Depends(get_db),
    registry: ModelRegistry = Depends(get_model_registry),
) -> BatchResponse[SentimentResponse]:
    """Run micro-batched sentiment analysis across multiple texts with partial-failure isolation."""
    service = BatchService(session=db, registry=registry)
    return await service.run_sentiment_batch(payload, developer=developer, api_key_id=current_key.id)


@router.post(
    "/ner",
    response_model=BatchResponse[NERResponse],
    status_code=status.HTTP_200_OK,
    summary="Batch Named Entity Recognition",
    dependencies=[
        Depends(require_scopes("ner:read", "nlp:read", "analyze:read", "admin")),
        Depends(check_rate_limit),
        Depends(check_quota),
    ],
)
async def batch_extract_entities(
    payload: BatchNERRequest,
    current_key: APIKey = Depends(get_current_api_key),
    developer: Developer = Depends(get_current_developer),
    db: AsyncSession = Depends(get_db),
    registry: ModelRegistry = Depends(get_model_registry),
) -> BatchResponse[NERResponse]:
    """Run micro-batched NER across multiple texts with character offset preservation."""
    service = BatchService(session=db, registry=registry)
    return await service.run_ner_batch(payload, developer=developer, api_key_id=current_key.id)


@router.post(
    "/classification",
    response_model=BatchResponse[ZeroShotResponse],
    status_code=status.HTTP_200_OK,
    summary="Batch zero-shot text classification",
    dependencies=[
        Depends(require_scopes("classification:read", "nlp:read", "analyze:read", "admin")),
        Depends(check_rate_limit),
        Depends(check_quota),
    ],
)
async def batch_classify_zero_shot(
    payload: BatchZeroShotRequest,
    current_key: APIKey = Depends(get_current_api_key),
    developer: Developer = Depends(get_current_developer),
    db: AsyncSession = Depends(get_db),
    registry: ModelRegistry = Depends(get_model_registry),
) -> BatchResponse[ZeroShotResponse]:
    """Run grouped batched zero-shot NLI classification across multiple texts."""
    service = BatchService(session=db, registry=registry)
    return await service.run_classification_batch(payload, developer=developer, api_key_id=current_key.id)


@router.post(
    "/keywords",
    response_model=BatchResponse[KeywordResponse],
    status_code=status.HTTP_200_OK,
    summary="Batch keyword extraction",
    dependencies=[
        Depends(require_scopes("keywords:read", "nlp:read", "analyze:read", "admin")),
        Depends(check_rate_limit),
        Depends(check_quota),
    ],
)
async def batch_extract_keywords(
    payload: BatchKeywordRequest,
    current_key: APIKey = Depends(get_current_api_key),
    developer: Developer = Depends(get_current_developer),
    db: AsyncSession = Depends(get_db),
    registry: ModelRegistry = Depends(get_model_registry),
) -> BatchResponse[KeywordResponse]:
    """Run embedding-based keyword extraction using deduplicated cross-document batched embeddings."""
    service = BatchService(session=db, registry=registry)
    return await service.run_keywords_batch(payload, developer=developer, api_key_id=current_key.id)


@router.post(
    "/summarization",
    response_model=BatchResponse[SummarizationResponse],
    status_code=status.HTTP_200_OK,
    summary="Batch abstractive text summarization",
    dependencies=[
        Depends(require_scopes("summarization:read", "nlp:read", "analyze:read", "admin")),
        Depends(check_rate_limit),
        Depends(check_quota),
    ],
)
async def batch_summarize_text(
    payload: BatchSummarizationRequest,
    current_key: APIKey = Depends(get_current_api_key),
    developer: Developer = Depends(get_current_developer),
    db: AsyncSession = Depends(get_db),
    registry: ModelRegistry = Depends(get_model_registry),
) -> BatchResponse[SummarizationResponse]:
    """Run bounded concurrent abstractive summarization with chunked consolidation per item."""
    service = BatchService(session=db, registry=registry)
    return await service.run_summarization_batch(payload, developer=developer, api_key_id=current_key.id)


@router.post(
    "/extraction",
    response_model=BatchResponse[ExtractionResponse],
    status_code=status.HTTP_200_OK,
    summary="Batch structured information extraction (local LLM)",
    dependencies=[
        Depends(require_scopes("extraction:read", "generate:read", "nlp:read", "analyze:read", "admin")),
        Depends(check_rate_limit),
        Depends(check_quota),
    ],
)
async def batch_extract_information(
    payload: BatchExtractionRequest,
    current_key: APIKey = Depends(get_current_api_key),
    developer: Developer = Depends(get_current_developer),
    db: AsyncSession = Depends(get_db),
    ollama_client: OllamaClient = Depends(get_ollama_client),
) -> BatchResponse[ExtractionResponse]:
    """Run bounded concurrent local LLM structured extraction with per-item schema validation."""
    service = BatchService(session=db, ollama_client=ollama_client)
    return await service.run_extraction_batch(payload, developer=developer, api_key_id=current_key.id)
