"""Pydantic schemas for M7 batch NLP processing requests and structured partial-failure responses."""

from __future__ import annotations

from typing import Any, Dict, Generic, List, Literal, Optional, Sequence, TypeVar, Union
from pydantic import BaseModel, Field, model_validator

from src.core.config import get_settings
from src.schemas.extraction import ExtractionField

ResultT = TypeVar("ResultT")


class _BatchRequestEnvelope(BaseModel):
    """Base envelope enforcing configurable minimum and maximum batch size bounds."""

    items: Sequence[Any]

    @model_validator(mode="after")
    def validate_batch_size(self) -> "_BatchRequestEnvelope":
        max_size = get_settings().BATCH_MAX_SIZE
        if len(self.items) < 1:
            raise ValueError("Batch 'items' must contain at least 1 item.")
        if len(self.items) > max_size:
            raise ValueError(
                f"Batch size {len(self.items)} exceeds maximum allowed batch size of {max_size}."
            )
        return self


class BatchItemError(BaseModel):
    """Safe, structured error metadata for an individual failed batch item."""

    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Safe human-readable error message")
    status_code: int = Field(..., description="HTTP status equivalent for this item error")


class BatchItemResult(BaseModel, Generic[ResultT]):
    """Per-item outcome preserving positional order and caller identifier."""

    index: int = Field(..., ge=0, description="Zero-based index in the submitted batch")
    id: Optional[str] = Field(default=None, description="Optional caller-provided item identifier")
    status: Literal["success", "error"] = Field(..., description="Execution status for this item")
    result: Optional[ResultT] = Field(default=None, description="Typed capability output on success")
    error: Optional[BatchItemError] = Field(default=None, description="Safe error payload on failure")
    cost_units: int = Field(default=0, ge=0, description="Metered cost units for this item")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Processing latency for this item in ms")


class BatchResponse(BaseModel, Generic[ResultT]):
    """Aggregate batch response containing ordered item results and metering summary."""

    capability: str = Field(..., description="NLP capability executed")
    execution_mode: str = Field(
        ...,
        description="Execution strategy used (e.g. batched_inference, deduplicated_batched_embeddings, bounded_concurrent)",
    )
    model_id: Optional[str] = Field(default=None, description="Primary model identifier used")
    total_items: int = Field(..., ge=1, description="Total items submitted in the request")
    succeeded_count: int = Field(..., ge=0, description="Count of items that succeeded")
    failed_count: int = Field(..., ge=0, description="Count of items that failed")
    total_cost_units: int = Field(..., ge=0, description="Total cost units metered across succeeded items")
    latency_ms: float = Field(..., ge=0.0, description="End-to-end batch execution latency in ms")
    results: List[BatchItemResult[ResultT]] = Field(
        ...,
        description="Per-item results in exact submitted order",
    )


# -----------------------------------------------------------------------------
# Batch Request Item & Request Schemas (6 Capabilities)
# -----------------------------------------------------------------------------


class BatchSentimentItem(BaseModel):
    """Single item for batch sentiment analysis."""

    id: Optional[str] = Field(default=None, max_length=128, description="Optional item identifier")
    text: str = Field(..., description="Text to analyze for sentiment")


class BatchSentimentRequest(_BatchRequestEnvelope):
    """Batch request for sentiment analysis."""

    items: List[BatchSentimentItem] = Field(..., min_length=1, description="Items to classify")


class BatchNERItem(BaseModel):
    """Single item for batch Named Entity Recognition."""

    id: Optional[str] = Field(default=None, max_length=128, description="Optional item identifier")
    text: str = Field(..., description="Text to scan for named entities")


class BatchNERRequest(_BatchRequestEnvelope):
    """Batch request for Named Entity Recognition."""

    items: List[BatchNERItem] = Field(..., min_length=1, description="Items to scan")


class BatchZeroShotItem(BaseModel):
    """Single item for batch zero-shot classification (supports item-level label overrides)."""

    id: Optional[str] = Field(default=None, max_length=128, description="Optional item identifier")
    text: str = Field(..., description="Text to classify")
    candidate_labels: Optional[List[str]] = Field(
        default=None,
        description="Optional per-item candidate labels overriding batch default",
    )
    multi_label: Optional[bool] = Field(
        default=None,
        description="Optional per-item multi_label flag overriding batch default",
    )


class BatchZeroShotRequest(_BatchRequestEnvelope):
    """Batch request for zero-shot classification."""

    items: List[BatchZeroShotItem] = Field(..., min_length=1, description="Items to classify")
    candidate_labels: Optional[List[str]] = Field(
        default=None,
        description="Default candidate labels applied to items without per-item labels",
    )
    multi_label: bool = Field(
        default=False,
        description="Default multi-label mode for items without per-item override",
    )


class BatchKeywordItem(BaseModel):
    """Single item for batch keyword extraction."""

    id: Optional[str] = Field(default=None, max_length=128, description="Optional item identifier")
    text: str = Field(..., description="Text to extract keywords from")
    max_keywords: Optional[int] = Field(default=None, description="Optional per-item max_keywords override")
    ngram_range: Optional[List[int]] = Field(default=None, description="Optional per-item [min_n, max_n] override")
    candidates: Optional[List[str]] = Field(default=None, description="Optional per-item candidate filter")


class BatchKeywordRequest(_BatchRequestEnvelope):
    """Batch request for embedding-based keyword extraction."""

    items: List[BatchKeywordItem] = Field(..., min_length=1, description="Items to analyze")
    max_keywords: int = Field(default=5, description="Default maximum keywords per item")
    ngram_range: List[int] = Field(default=[1, 1], description="Default n-gram range [min_n, max_n]")
    candidates: Optional[List[str]] = Field(default=None, description="Default candidate phrase filter")


class BatchSummarizationItem(BaseModel):
    """Single item for batch abstractive summarization."""

    id: Optional[str] = Field(default=None, max_length=128, description="Optional item identifier")
    text: str = Field(..., description="Document text to summarize")
    min_length: Optional[int] = Field(default=None, description="Optional per-item min_length override")
    max_length: Optional[int] = Field(default=None, description="Optional per-item max_length override")
    do_sample: Optional[bool] = Field(default=None, description="Optional per-item do_sample override")
    temperature: Optional[float] = Field(default=None, description="Optional per-item temperature override")


class BatchSummarizationRequest(_BatchRequestEnvelope):
    """Batch request for abstractive text summarization."""

    items: List[BatchSummarizationItem] = Field(..., min_length=1, description="Documents to summarize")
    min_length: int = Field(default=30, description="Default minimum target token length")
    max_length: int = Field(default=130, description="Default maximum target token length")
    do_sample: bool = Field(default=False, description="Default sampling flag")
    temperature: Optional[float] = Field(default=None, description="Default sampling temperature")


class BatchExtractionItem(BaseModel):
    """Single item for batch structured information extraction."""

    id: Optional[str] = Field(default=None, max_length=128, description="Optional item identifier")
    text: str = Field(..., description="Source document text")
    schema_definition: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional per-item JSON Schema override",
    )
    fields: Optional[List[Union[ExtractionField, Dict[str, Any]]]] = Field(
        default=None,
        description="Optional per-item field definitions override",
    )
    instructions: Optional[str] = Field(default=None, description="Optional per-item extraction instructions")
    model: Optional[str] = Field(default=None, description="Optional per-item Ollama model override")
    temperature: Optional[float] = Field(default=None, description="Optional per-item temperature override")


class BatchExtractionRequest(_BatchRequestEnvelope):
    """Batch request for local LLM structured information extraction."""

    items: List[BatchExtractionItem] = Field(..., min_length=1, description="Documents to extract")
    schema_definition: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Default JSON Schema specification for batch items",
    )
    fields: Optional[List[Union[ExtractionField, Dict[str, Any]]]] = Field(
        default=None,
        description="Default simplified field list for batch items",
    )
    instructions: Optional[str] = Field(default=None, description="Default extraction instructions")
    model: Optional[str] = Field(default=None, description="Default Ollama model override")
    temperature: float = Field(default=0.0, description="Default LLM sampling temperature")
