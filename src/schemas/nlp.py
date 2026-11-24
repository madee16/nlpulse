"""Pydantic schemas for NLP inference requests and structured response payloads."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


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
