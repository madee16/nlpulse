"""Pydantic schemas for NLP inference requests and structured response payloads."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


# -----------------------------------------------------------------------------
# Sentiment Analysis
# -----------------------------------------------------------------------------

class SentimentRequest(BaseModel):
    """Payload for text sentiment analysis."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text content to evaluate for sentiment polarity",
        examples=["NLPulse provides exceptionally fast inference with robust error handling."],
    )


class SentimentResponse(BaseModel):
    """Structured response from sentiment analysis."""

    label: str = Field(..., description="Predicted top sentiment label (e.g. POSITIVE, NEGATIVE)")
    confidence: float = Field(..., description="Probability confidence of top prediction (0.0 to 1.0)")
    scores: Dict[str, float] = Field(..., description="Full probability distribution across all model classes")
    model_id: str = Field(..., description="Model identifier utilized for inference")
    latency_ms: float = Field(..., description="Inference execution latency in milliseconds")


# -----------------------------------------------------------------------------
# Named Entity Recognition (NER)
# -----------------------------------------------------------------------------

class EntityItem(BaseModel):
    """An individual extracted named entity span."""

    entity: str = Field(..., description="Entity category/tag (e.g. PER, ORG, LOC, MISC)")
    text: str = Field(..., description="Exact textual span from input document")
    confidence: float = Field(..., description="Confidence probability for this entity (0.0 to 1.0)")
    start_offset: int = Field(..., description="Zero-based character index start offset")
    end_offset: int = Field(..., description="Zero-based character index end offset")


class NERRequest(BaseModel):
    """Payload for Named Entity Recognition extraction."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text document to analyze for named entities",
        examples=["Sundar Pichai announced new AI infrastructure in Mountain View, California."],
    )


class NERResponse(BaseModel):
    """Structured response containing all detected named entity spans."""

    entities: List[EntityItem] = Field(..., description="Extracted entity spans with classification and offsets")
    entity_count: int = Field(..., description="Total count of extracted entities")
    model_id: str = Field(..., description="Model identifier utilized for inference")
    latency_ms: float = Field(..., description="Inference execution latency in milliseconds")


# -----------------------------------------------------------------------------
# Zero-Shot Text Classification
# -----------------------------------------------------------------------------

class LabelScore(BaseModel):
    """Single candidate label and its predicted probability score."""

    label: str = Field(..., description="Candidate classification label")
    score: float = Field(..., description="Predicted probability score")


class ZeroShotRequest(BaseModel):
    """Payload for NLI-based zero-shot text classification."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Document or passage to classify",
        examples=["Apple reported record quarterly revenue driven by cloud services and hardware sales."],
    )
    candidate_labels: List[str] = Field(
        ...,
        min_length=1,
        description="Candidate category labels for zero-shot hypothesis testing",
        examples=[["technology", "finance", "healthcare", "politics"]],
    )
    multi_label: bool = Field(
        default=False,
        description="If True, evaluates independent probabilities per label; if False, applies softmax across all labels",
    )


class ZeroShotResponse(BaseModel):
    """Structured response from zero-shot classification with ranking."""

    top_label: Optional[str] = Field(None, description="Highest scoring candidate label")
    top_score: Optional[float] = Field(None, description="Score of the highest scoring candidate label")
    predictions: List[LabelScore] = Field(..., description="Sorted predictions in descending score order")
    multi_label: bool = Field(..., description="Classification probability mode employed")
    model_id: str = Field(..., description="Model identifier utilized for inference")
    latency_ms: float = Field(..., description="Inference execution latency in milliseconds")


# -----------------------------------------------------------------------------
# Keyword & Keyphrase Extraction (M5)
# -----------------------------------------------------------------------------

class KeywordItem(BaseModel):
    """An individual extracted keyword or keyphrase with relevance score."""

    keyword: str = Field(..., description="Extracted keyword or keyphrase")
    score: float = Field(..., description="Cosine similarity relevance score against document embedding")


class KeywordRequest(BaseModel):
    """Payload for embedding-based keyword and keyphrase extraction."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text document to analyze for key concepts",
        examples=["FastAPI provides automatic OpenAPI documentation and high asynchronous throughput."],
    )
    max_keywords: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of top keywords to return",
    )
    ngram_range: List[int] = Field(
        default=[1, 1],
        min_length=2,
        max_length=2,
        description="N-gram length bounds [min_n, max_n] for candidate phrases (e.g. [1, 2] for unigrams and bigrams)",
    )
    candidates: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific candidate phrases to filter and rank",
    )


class KeywordResponse(BaseModel):
    """Structured response containing ranked keywords and extraction telemetry."""

    keywords: List[KeywordItem] = Field(..., description="Ranked keywords in descending relevance order")
    total_candidates: int = Field(..., description="Total candidate n-grams evaluated")
    model_id: str = Field(..., description="Embedding model identifier utilized for extraction")
    latency_ms: float = Field(..., description="Inference execution latency in milliseconds")


# -----------------------------------------------------------------------------
# Abstractive Text Summarization (M5)
# -----------------------------------------------------------------------------

class SummarizationRequest(BaseModel):
    """Payload for abstractive document summarization."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Document or passage to summarize",
        examples=["PostgreSQL 16 introduces enhanced logical replication and query execution optimizations..."],
    )
    min_length: int = Field(
        default=30,
        ge=5,
        le=500,
        description="Minimum target length in tokens for generated summary",
    )
    max_length: int = Field(
        default=130,
        ge=10,
        le=1000,
        description="Maximum target length in tokens for generated summary",
    )
    do_sample: bool = Field(
        default=False,
        description="Whether to use multinomial sampling instead of greedy beam search",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature if do_sample is True",
    )

    @model_validator(mode="after")
    def validate_generation_params(self) -> "SummarizationRequest":
        if self.min_length > self.max_length:
            raise ValueError(f"min_length ({self.min_length}) cannot exceed max_length ({self.max_length}).")
        if not self.do_sample and self.temperature is not None:
            raise ValueError("temperature is only effective when do_sample=True. Set do_sample=True or omit temperature.")
        if self.do_sample and self.temperature is not None and self.temperature <= 0.0:
            raise ValueError("temperature must be strictly greater than 0.0 when do_sample=True.")
        return self


class SummarizationResponse(BaseModel):
    """Structured response containing generated summary and chunking telemetry."""

    summary: str = Field(..., description="Coherent abstractive summary text")
    original_character_count: int = Field(..., description="Original input document character count")
    summary_character_count: int = Field(..., description="Generated summary character count")
    chunks_processed: int = Field(..., description="Total document chunks processed and consolidated")
    model_id: str = Field(..., description="Summarization model identifier utilized")
    latency_ms: float = Field(..., description="Total inference latency in milliseconds")
