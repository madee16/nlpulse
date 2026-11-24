"""Integration tests for Core NLP API endpoints: Sentiment, NER, and Zero-Shot Classification."""

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from src.core.config import get_settings
from src.core.models.base import ModelMetadata
from src.core.models.nlp import NERModel, SentimentModel, ZeroShotClassificationModel
from src.core.rate_limiter import rate_limiter
from src.db.session import get_db
from src.main import app
from src.models.developer import Developer
from src.models.usage_record import UsageRecord
from src.services.model_registry import ModelRegistry

settings = get_settings()


@pytest.fixture(autouse=True)
async def reset_limiter_and_mock_models():
    """Reset rate limiter and ensure test models with mock pipelines are loaded."""
    await rate_limiter.reset()

    # Create mock registry in app.state
    reg = ModelRegistry()

    # Sentiment mock
    s_meta = ModelMetadata(
        model_id=settings.MODEL_SENTIMENT_ID,
        name="Sentiment Test",
        task="sentiment",
    )
    def mock_sentiment(text: str):
        if "great" in text.lower() or "love" in text.lower():
            return [{"label": "POSITIVE", "score": 0.9921}, {"label": "NEGATIVE", "score": 0.0079}]
        return [{"label": "NEGATIVE", "score": 0.9634}, {"label": "POSITIVE", "score": 0.0366}]

    reg.register(s_meta, lambda: SentimentModel(s_meta, pipeline_factory=lambda: mock_sentiment))

    # NER mock
    ner_meta = ModelMetadata(
        model_id=settings.MODEL_NER_ID,
        name="NER Test",
        task="ner",
    )
    def mock_ner(text: str):
        if "sundar" in text.lower():
            return [{"entity_group": "PER", "score": 0.99, "word": "Sundar Pichai", "start": 0, "end": 13}]
        return []

    reg.register(ner_meta, lambda: NERModel(ner_meta, pipeline_factory=lambda: mock_ner))

    # Zero-shot mock
    zs_meta = ModelMetadata(
        model_id=settings.MODEL_ZERO_SHOT_ID,
        name="Zero Shot Test",
        task="zero-shot",
    )
    def mock_zero_shot(text: str, candidate_labels: list, multi_label: bool):
        scores = [round(0.8 / (i + 1), 4) for i in range(len(candidate_labels))]
        return {"sequence": text, "labels": candidate_labels, "scores": scores}

    reg.register(zs_meta, lambda: ZeroShotClassificationModel(zs_meta, pipeline_factory=lambda: mock_zero_shot))

    app.state.model_registry = reg


@pytest.fixture
async def registered_auth_headers(async_client: AsyncClient):
    """Register developer and return authorization headers with a full-scoped key."""
    res = await async_client.post(
        "/api/v1/developers",
        json={"email": "nlp_test_dev@example.com", "name": "NLP Dev"},
    )
    assert res.status_code == 201
    data = res.json()
    return {"X-API-Key": data["api_key"]}, data["id"]


# -----------------------------------------------------------------------------
# Sentiment Analysis API Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sentiment_api_success(async_client: AsyncClient, registered_auth_headers: tuple):
    """Verify POST /api/v1/nlp/sentiment returns 200 and records usage in database."""
    headers, dev_id = registered_auth_headers

    response = await async_client.post(
        "/api/v1/nlp/sentiment",
        headers=headers,
        json={"text": "NLPulse is a great and highly reliable service!"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "POSITIVE"
    assert data["confidence"] > 0.9
    assert "POSITIVE" in data["scores"]
    assert data["latency_ms"] >= 0.0
    assert data["model_id"] == settings.MODEL_SENTIMENT_ID


@pytest.mark.asyncio
async def test_sentiment_api_unauthenticated(async_client: AsyncClient):
    """Verify POST /api/v1/nlp/sentiment requires authentication."""
    response = await async_client.post(
        "/api/v1/nlp/sentiment",
        json={"text": "Unauthenticated test"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_sentiment_api_insufficient_scope(async_client: AsyncClient, registered_auth_headers: tuple):
    """Verify key lacking sentiment scope receives 403 Forbidden."""
    headers, _ = registered_auth_headers

    # Create restricted child key lacking sentiment:read
    key_res = await async_client.post(
        "/api/v1/keys",
        headers=headers,
        json={"name": "Restricted", "scopes": ["usage:read"]},
    )
    assert key_res.status_code == 201
    restricted_key = key_res.json()["api_key"]

    response = await async_client.post(
        "/api/v1/nlp/sentiment",
        headers={"X-API-Key": restricted_key},
        json={"text": "Testing restricted key"},
    )
    assert response.status_code == 403
    assert "lacks required scope" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_sentiment_api_empty_text_validation(async_client: AsyncClient, registered_auth_headers: tuple):
    """Verify empty text fails with 422 Unprocessable Entity."""
    headers, _ = registered_auth_headers

    response = await async_client.post(
        "/api/v1/nlp/sentiment",
        headers=headers,
        json={"text": ""},
    )
    assert response.status_code == 422


# -----------------------------------------------------------------------------
# Named Entity Recognition API Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ner_api_success(async_client: AsyncClient, registered_auth_headers: tuple):
    """Verify POST /api/v1/nlp/ner extracts named entity spans with offsets."""
    headers, _ = registered_auth_headers

    response = await async_client.post(
        "/api/v1/nlp/ner",
        headers=headers,
        json={"text": "Sundar announced new models."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["entity_count"] == 1
    assert data["entities"][0]["entity"] == "PER"
    assert data["entities"][0]["text"] == "Sundar Pichai"
    assert data["entities"][0]["start_offset"] == 0


@pytest.mark.asyncio
async def test_ner_api_entity_free_text(async_client: AsyncClient, registered_auth_headers: tuple):
    """Verify text without named entities returns empty list and entity_count=0."""
    headers, _ = registered_auth_headers

    response = await async_client.post(
        "/api/v1/nlp/ner",
        headers=headers,
        json={"text": "Everything is running smoothly without names."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["entity_count"] == 0
    assert data["entities"] == []


# -----------------------------------------------------------------------------
# Zero-Shot Classification API Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_zero_shot_api_single_label(async_client: AsyncClient, registered_auth_headers: tuple):
    """Verify POST /api/v1/nlp/classification executes single-label zero-shot classification."""
    headers, _ = registered_auth_headers

    response = await async_client.post(
        "/api/v1/nlp/classification",
        headers=headers,
        json={
            "text": "The government passed a major healthcare reform bill.",
            "candidate_labels": ["politics", "sports", "entertainment"],
            "multi_label": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["top_label"] == "politics"
    assert data["multi_label"] is False
    assert len(data["predictions"]) == 3
    assert data["predictions"][0]["label"] == "politics"


@pytest.mark.asyncio
async def test_zero_shot_api_multi_label(async_client: AsyncClient, registered_auth_headers: tuple):
    """Verify zero-shot classification in multi_label mode."""
    headers, _ = registered_auth_headers

    response = await async_client.post(
        "/api/v1/nlp/classification",
        headers=headers,
        json={
            "text": "Medical research institute secures venture funding for genomic sequencing.",
            "candidate_labels": ["healthcare", "finance", "technology"],
            "multi_label": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["multi_label"] is True
    assert len(data["predictions"]) == 3


@pytest.mark.asyncio
async def test_zero_shot_api_empty_labels_rejected(async_client: AsyncClient, registered_auth_headers: tuple):
    """Verify zero-shot classification rejects empty candidate labels."""
    headers, _ = registered_auth_headers

    response = await async_client.post(
        "/api/v1/nlp/classification",
        headers=headers,
        json={
            "text": "Some valid passage to classify",
            "candidate_labels": [],
        },
    )
    assert response.status_code == 422


# -----------------------------------------------------------------------------
# Quota Exhaustion Integration Test
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nlp_api_quota_exhaustion_blocks_inference(
    async_client: AsyncClient,
    registered_auth_headers: tuple,
):
    """Verify exhausted developer quota blocks NLP inference with 429."""
    headers, dev_id = registered_auth_headers

    import uuid

    # Manually exhaust the developer's monthly quota by setting monthly_request_quota to 0
    # or updating developer quota directly
    get_db_override = app.dependency_overrides[get_db]
    async for db in get_db_override():
        dev = await db.get(Developer, uuid.UUID(dev_id))
        dev.monthly_request_quota = 0
        await db.commit()
        break

    response = await async_client.post(
        "/api/v1/nlp/sentiment",
        headers=headers,
        json={"text": "Will be blocked by quota"},
    )
    assert response.status_code == 429
    detail = response.json()["detail"].lower()
    assert "quota" in detail and "exhausted" in detail
