"""Unit tests for M7 batch request and response Pydantic schemas."""

import pytest
from pydantic import ValidationError

from src.core.config import get_settings
from src.schemas.batch import (
    BatchExtractionRequest,
    BatchItemError,
    BatchItemResult,
    BatchKeywordRequest,
    BatchNERRequest,
    BatchResponse,
    BatchSentimentRequest,
    BatchSummarizationRequest,
    BatchZeroShotRequest,
)
from src.schemas.nlp import SentimentResponse

settings = get_settings()


@pytest.mark.parametrize(
    "schema_cls",
    [
        BatchSentimentRequest,
        BatchNERRequest,
        BatchZeroShotRequest,
        BatchKeywordRequest,
        BatchSummarizationRequest,
        BatchExtractionRequest,
    ],
)
def test_batch_request_rejects_empty_items(schema_cls):
    """Verify all six batch request schemas reject empty items lists."""
    with pytest.raises(ValidationError):
        schema_cls(items=[])


def test_batch_request_rejects_oversized_items_dynamically(monkeypatch):
    """Verify batch request rejects items exceeding configured BATCH_MAX_SIZE."""
    monkeypatch.setattr(settings, "BATCH_MAX_SIZE", 2)

    valid = BatchSentimentRequest(items=[{"text": "one"}, {"text": "two"}])
    assert len(valid.items) == 2

    with pytest.raises(ValidationError, match="exceeds maximum allowed batch size"):
        BatchSentimentRequest(items=[{"text": "one"}, {"text": "two"}, {"text": "three"}])


def test_batch_request_allows_invalid_individual_item_for_partial_failure():
    """Verify empty/invalid individual item texts are accepted into envelope for isolated per-item failure."""
    req = BatchSentimentRequest(
        items=[
            {"id": "ok-1", "text": "Great experience"},
            {"id": "bad-1", "text": ""},
            {"id": "bad-2", "text": "   "},
        ]
    )
    assert len(req.items) == 3
    assert req.items[0].id == "ok-1"
    assert req.items[1].text == ""


def test_batch_request_shared_defaults_and_per_item_overrides():
    """Verify batch-level defaults and per-item overrides across classification, keywords, summarization, extraction."""
    zs_req = BatchZeroShotRequest(
        candidate_labels=["tech", "finance"],
        multi_label=True,
        items=[
            {"id": "1", "text": "Cloud revenue surged."},
            {"id": "2", "text": "Goal scored!", "candidate_labels": ["sports", "news"], "multi_label": False},
        ],
    )
    assert zs_req.candidate_labels == ["tech", "finance"]
    assert zs_req.items[1].candidate_labels == ["sports", "news"]
    assert zs_req.items[1].multi_label is False

    kw_req = BatchKeywordRequest(
        max_keywords=3,
        ngram_range=[1, 2],
        items=[{"text": "FastAPI is fast."}],
    )
    assert kw_req.max_keywords == 3
    assert kw_req.ngram_range == [1, 2]

    sum_req = BatchSummarizationRequest(
        min_length=20,
        max_length=100,
        items=[{"text": "Long document text."}],
    )
    assert sum_req.min_length == 20

    ext_req = BatchExtractionRequest(
        fields=[{"name": "company", "type": "string"}],
        items=[{"id": "doc-1", "text": "Google launched Gemini."}],
    )
    assert ext_req.fields is not None
    assert ext_req.items[0].id == "doc-1"


def test_batch_response_preserves_order_and_mixed_results():
    """Verify BatchResponse carries aggregate counts and typed/error items in order."""
    resp = BatchResponse[SentimentResponse](
        capability="sentiment",
        execution_mode="batched_inference",
        model_id="distilbert-sst2",
        total_items=2,
        succeeded_count=1,
        failed_count=1,
        total_cost_units=1,
        latency_ms=12.5,
        results=[
            BatchItemResult[SentimentResponse](
                index=0,
                id="item-a",
                status="success",
                cost_units=1,
                latency_ms=10.0,
                result=SentimentResponse(
                    label="POSITIVE",
                    confidence=0.99,
                    scores={"POSITIVE": 0.99, "NEGATIVE": 0.01},
                    model_id="distilbert-sst2",
                    latency_ms=10.0,
                ),
            ),
            BatchItemResult[SentimentResponse](
                index=1,
                id="item-b",
                status="error",
                cost_units=0,
                latency_ms=0.5,
                error=BatchItemError(
                    code="validation_error",
                    message="Input text cannot be empty or solely whitespace.",
                    status_code=422,
                ),
            ),
        ],
    )
    assert resp.succeeded_count == 1
    assert resp.failed_count == 1
    assert resp.results[0].index == 0
    assert resp.results[1].error.status_code == 422
