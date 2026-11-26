"""API endpoints for Core NLP pipelines: Sentiment Analysis, NER, and Zero-Shot Classification."""

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import (
    check_rate_limit,
    get_current_api_key,
    get_current_developer,
    get_model_registry,
    get_ollama_client,
    require_scopes,
    reserve_single_quota,
)
from src.db.session import get_db
from src.models.api_key import APIKey
from src.models.developer import Developer
from src.schemas.extraction import ExtractionRequest, ExtractionResponse
from src.schemas.nlp import (
    EntityItem,
    KeywordRequest,
    KeywordResponse,
    NERRequest,
    NERResponse,
    SentimentRequest,
    SentimentResponse,
    SummarizationRequest,
    SummarizationResponse,
    ZeroShotRequest,
    ZeroShotResponse,
)
from src.services.extraction_service import ExtractionService
from src.services.model_registry import ModelRegistry
from src.services.nlp_service import NLPService
from src.services.ollama_client import OllamaClient

router = APIRouter(prefix="/nlp", tags=["NLP Pipelines (M4-M6)"])


@router.post(
    "/sentiment",
    response_model=SentimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze text sentiment polarity",
    dependencies=[
        Depends(require_scopes("sentiment:read", "nlp:read", "analyze:read", "admin")),
        Depends(check_rate_limit),
        Depends(reserve_single_quota),
    ],
)
async def analyze_sentiment(
    payload: SentimentRequest,
    current_key: APIKey = Depends(get_current_api_key),
    developer: Developer = Depends(get_current_developer),
    db: AsyncSession = Depends(get_db),
    registry: ModelRegistry = Depends(get_model_registry),
) -> SentimentResponse:
    """Evaluate text sentiment, returning predicted polarity label, confidence score, and distribution."""
    service = NLPService(session=db, registry=registry)
    result = await service.analyze_sentiment(
        text=payload.text,
        developer_id=developer.id,
        api_key_id=current_key.id,
    )
    return SentimentResponse(**result)


@router.post(
    "/ner",
    response_model=NERResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract named entities with offsets",
    dependencies=[
        Depends(require_scopes("ner:read", "nlp:read", "analyze:read", "admin")),
        Depends(check_rate_limit),
        Depends(reserve_single_quota),
    ],
)
async def extract_entities(
    payload: NERRequest,
    current_key: APIKey = Depends(get_current_api_key),
    developer: Developer = Depends(get_current_developer),
    db: AsyncSession = Depends(get_db),
    registry: ModelRegistry = Depends(get_model_registry),
) -> NERResponse:
    """Identify named entity spans (persons, locations, organizations) with character offsets."""
    service = NLPService(session=db, registry=registry)
    result = await service.extract_entities(
        text=payload.text,
        developer_id=developer.id,
        api_key_id=current_key.id,
    )
    return NERResponse(**result)


@router.post(
    "/classification",
    response_model=ZeroShotResponse,
    status_code=status.HTTP_200_OK,
    summary="Zero-shot text classification against candidate labels",
    dependencies=[
        Depends(require_scopes("classification:read", "nlp:read", "analyze:read", "admin")),
        Depends(check_rate_limit),
        Depends(reserve_single_quota),
    ],
)
async def classify_zero_shot(
    payload: ZeroShotRequest,
    current_key: APIKey = Depends(get_current_api_key),
    developer: Developer = Depends(get_current_developer),
    db: AsyncSession = Depends(get_db),
    registry: ModelRegistry = Depends(get_model_registry),
) -> ZeroShotResponse:
    """Classify text into arbitrary user-provided candidate labels using NLI without task-specific training."""
    service = NLPService(session=db, registry=registry)
    result = await service.classify_zero_shot(
        text=payload.text,
        candidate_labels=payload.candidate_labels,
        multi_label=payload.multi_label,
        developer_id=developer.id,
        api_key_id=current_key.id,
    )
    return ZeroShotResponse(**result)


@router.post(
    "/keywords",
    response_model=KeywordResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract and rank key concepts and n-grams using dense embeddings",
    dependencies=[
        Depends(require_scopes("keywords:read", "nlp:read", "analyze:read", "admin")),
        Depends(check_rate_limit),
        Depends(reserve_single_quota),
    ],
)
async def extract_keywords(
    payload: KeywordRequest,
    current_key: APIKey = Depends(get_current_api_key),
    developer: Developer = Depends(get_current_developer),
    db: AsyncSession = Depends(get_db),
    registry: ModelRegistry = Depends(get_model_registry),
) -> KeywordResponse:
    """Extract key terms and keyphrases from input text using embedding cosine similarity."""
    service = NLPService(session=db, registry=registry)
    result = await service.extract_keywords(
        text=payload.text,
        max_keywords=payload.max_keywords,
        ngram_range=(payload.ngram_range[0], payload.ngram_range[1]),
        candidates=payload.candidates,
        developer_id=developer.id,
        api_key_id=current_key.id,
    )
    return KeywordResponse(**result)


@router.post(
    "/summarization",
    response_model=SummarizationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate abstractive summary of input document",
    dependencies=[
        Depends(require_scopes("summarization:read", "nlp:read", "analyze:read", "admin")),
        Depends(check_rate_limit),
        Depends(reserve_single_quota),
    ],
)
async def summarize_text(
    payload: SummarizationRequest,
    current_key: APIKey = Depends(get_current_api_key),
    developer: Developer = Depends(get_current_developer),
    db: AsyncSession = Depends(get_db),
    registry: ModelRegistry = Depends(get_model_registry),
) -> SummarizationResponse:
    """Generate a concise, coherent abstractive summary, handling long text via chunked consolidation."""
    service = NLPService(session=db, registry=registry)
    result = await service.summarize_text(
        text=payload.text,
        min_length=payload.min_length,
        max_length=payload.max_length,
        do_sample=payload.do_sample,
        temperature=payload.temperature,
        developer_id=developer.id,
        api_key_id=current_key.id,
    )
    return SummarizationResponse(**result)


@router.post(
    "/extraction",
    response_model=ExtractionResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract grounded structured information via local LLM (M6)",
    description=(
        "Extracts structured JSON adhering to a user-provided schema using a local generative model. "
        "Strictly enforces factual grounding without hallucinations, populating unstated fields with null."
    ),
    dependencies=[
        Depends(require_scopes("extraction:read", "generate:read", "nlp:read", "analyze:read", "admin")),
        Depends(check_rate_limit),
        Depends(reserve_single_quota),
    ],
)
async def extract_information(
    payload: ExtractionRequest,
    current_key: APIKey = Depends(get_current_api_key),
    developer: Developer = Depends(get_current_developer),
    db: AsyncSession = Depends(get_db),
    ollama_client: OllamaClient = Depends(get_ollama_client),
) -> ExtractionResponse:
    """Extract structured, schema-compliant JSON data from source text using local generative model."""
    service = ExtractionService(session=db, ollama_client=ollama_client)
    result = await service.extract_structured_data(
        text=payload.text,
        developer_id=developer.id,
        schema_definition=payload.schema_definition,
        fields=payload.fields,
        instructions=payload.instructions,
        model=payload.model,
        temperature=payload.temperature,
        api_key_id=current_key.id,
    )
    return ExtractionResponse(**result)
