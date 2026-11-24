"""Integration tests for M5 Advanced NLP API endpoints: Keywords and Summarization."""

import uuid
import numpy as np
import pytest
from httpx import AsyncClient
from sqlalchemy import select

from src.core.config import get_settings
from src.core.models.base import ModelMetadata
from src.core.models.nlp import EmbeddingModel, SummarizationModel
from src.core.rate_limiter import rate_limiter
from src.main import app
from src.models.developer import Developer
from src.models.usage_record import UsageRecord
from src.services.model_registry import ModelRegistry

settings = get_settings()


@pytest.fixture(autouse=True)
async def reset_limiter_and_mock_m5_models():
    """Reset rate limiter and ensure test models with mock pipelines are loaded."""
    await rate_limiter.reset()

    reg = ModelRegistry()

    # Embedding mock
    emb_meta = ModelMetadata(
        model_id=settings.MODEL_EMBEDDING_ID,
        name="Test Embedding",
        task="embedding",
    )
    def mock_emb(texts: list):
        res = []
        for t in texts:
            if "database" in t.lower() or "sql" in t.lower():
                vec = np.array([0.9, 0.1, 0.0], dtype=np.float32)
            elif "fastapi" in t.lower() or "api" in t.lower():
                vec = np.array([0.1, 0.9, 0.0], dtype=np.float32)
            else:
                vec = np.array([0.3, 0.3, 0.3], dtype=np.float32)
            res.append(vec / np.linalg.norm(vec))
        return np.array(res, dtype=np.float32)

    reg.register(emb_meta, lambda: EmbeddingModel(emb_meta, pipeline_factory=lambda: mock_emb))

    # Summarization mock
    sum_meta = ModelMetadata(
        model_id=settings.MODEL_SUMMARIZATION_ID,
        name="Test Summarization",
        task="summarization",
    )
    def mock_sum(text: str, **kwargs):
        return [{"summary_text": f"Abstractive summary of: {text[:35]}..."}]

    reg.register(sum_meta, lambda: SummarizationModel(sum_meta, pipeline_factory=lambda: mock_sum))

    app.state.model_registry = reg


@pytest.fixture
async def registered_dev_headers(async_client: AsyncClient):
    """Register developer and return authorization headers with a full-scoped key."""
    res = await async_client.post(
        "/api/v1/developers",
        json={"email": "m5_dev@example.com", "name": "M5 Advanced Dev"},
    )
    assert res.status_code == 201
    data = res.json()
    return {"X-API-Key": data["api_key"]}, data["id"]


# -----------------------------------------------------------------------------
# Keyword Extraction API Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_keywords_api_success(async_client: AsyncClient, registered_dev_headers: tuple, db_session):
    """Verify POST /api/v1/nlp/keywords extracts keywords and creates UsageRecord."""
    headers, dev_id = registered_dev_headers

    response = await async_client.post(
        "/api/v1/nlp/keywords",
        headers=headers,
        json={
            "text": "FastAPI enables high performance web APIs and microservices in Python.",
            "max_keywords": 3,
            "ngram_range": [1, 2],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == settings.MODEL_EMBEDDING_ID
    assert len(data["keywords"]) <= 3
    assert data["total_candidates"] >= 3
    assert data["latency_ms"] >= 0.0

    # Verify DB usage record
    res = await db_session.execute(
        select(UsageRecord).where(UsageRecord.developer_id == uuid.UUID(dev_id))
    )
    records = res.scalars().all()
    assert len(records) >= 1
    kw_records = [r for r in records if "/nlp/keywords" in r.endpoint]
    assert len(kw_records) == 1
    assert kw_records[0].cost_units == 1


@pytest.mark.asyncio
async def test_keywords_api_unauthenticated(async_client: AsyncClient):
    """Verify POST /api/v1/nlp/keywords without API key returns 401."""
    response = await async_client.post(
        "/api/v1/nlp/keywords",
        json={"text": "Valid text."},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_keywords_api_insufficient_scope(async_client: AsyncClient, registered_dev_headers: tuple):
    """Verify API key lacking keywords:read receives 403 Forbidden."""
    headers, _ = registered_dev_headers

    # Issue child key with only sentiment:read
    key_res = await async_client.post(
        "/api/v1/keys",
        headers=headers,
        json={"name": "Sentiment-Only Key", "scopes": ["sentiment:read"]},
    )
    assert key_res.status_code == 201
    limited_key = key_res.json()["api_key"]

    response = await async_client.post(
        "/api/v1/nlp/keywords",
        headers={"X-API-Key": limited_key},
        json={"text": "Valid text."},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_keywords_api_validation_error(async_client: AsyncClient, registered_dev_headers: tuple):
    """Verify empty text or invalid ngram_range returns 422."""
    headers, _ = registered_dev_headers

    # Empty text
    res = await async_client.post(
        "/api/v1/nlp/keywords",
        headers=headers,
        json={"text": ""},
    )
    assert res.status_code == 422

    # Invalid ngram_range bounds
    res2 = await async_client.post(
        "/api/v1/nlp/keywords",
        headers=headers,
        json={"text": "Valid text", "ngram_range": [3, 1]},
    )
    assert res2.status_code == 422


# -----------------------------------------------------------------------------
# Text Summarization API Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summarization_api_success(async_client: AsyncClient, registered_dev_headers: tuple, db_session):
    """Verify POST /api/v1/nlp/summarization generates summary and records usage."""
    headers, dev_id = registered_dev_headers

    text = "PostgreSQL is an open-source object-relational database system with proven reliability and integrity."
    response = await async_client.post(
        "/api/v1/nlp/summarization",
        headers=headers,
        json={
            "text": text,
            "min_length": 10,
            "max_length": 60,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == settings.MODEL_SUMMARIZATION_ID
    assert "Abstractive summary" in data["summary"]
    assert data["original_character_count"] == len(text)
    assert data["chunks_processed"] == 1
    assert data["latency_ms"] >= 0.0

    # Verify DB usage record
    res = await db_session.execute(
        select(UsageRecord).where(UsageRecord.developer_id == uuid.UUID(dev_id))
    )
    records = res.scalars().all()
    sum_records = [r for r in records if "/nlp/summarization" in r.endpoint]
    assert len(sum_records) == 1
    assert sum_records[0].cost_units == 1


@pytest.mark.asyncio
async def test_summarization_api_unauthenticated(async_client: AsyncClient):
    """Verify POST /api/v1/nlp/summarization without API key returns 401."""
    response = await async_client.post(
        "/api/v1/nlp/summarization",
        json={"text": "Valid text."},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_summarization_api_insufficient_scope(async_client: AsyncClient, registered_dev_headers: tuple):
    """Verify API key lacking summarization:read receives 403 Forbidden."""
    headers, _ = registered_dev_headers

    key_res = await async_client.post(
        "/api/v1/keys",
        headers=headers,
        json={"name": "NER-Only Key", "scopes": ["ner:read"]},
    )
    assert key_res.status_code == 201
    limited_key = key_res.json()["api_key"]

    response = await async_client.post(
        "/api/v1/nlp/summarization",
        headers={"X-API-Key": limited_key},
        json={"text": "Valid text to summarize."},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_summarization_api_invalid_lengths(async_client: AsyncClient, registered_dev_headers: tuple):
    """Verify min_length > max_length returns 422 Unprocessable Entity."""
    headers, _ = registered_dev_headers

    response = await async_client.post(
        "/api/v1/nlp/summarization",
        headers=headers,
        json={"text": "Valid text", "min_length": 80, "max_length": 30},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_m5_api_quota_exhaustion_blocks(async_client: AsyncClient, registered_dev_headers: tuple, db_session):
    """Verify exhausted monthly quota immediately returns 429 for both keywords and summarization."""
    headers, dev_id = registered_dev_headers

    dev = await db_session.get(Developer, uuid.UUID(dev_id))
    assert dev is not None
    dev.monthly_request_quota = 0
    await db_session.commit()

    # Keywords blocked
    kw_res = await async_client.post(
        "/api/v1/nlp/keywords",
        headers=headers,
        json={"text": "FastAPI is fast."},
    )
    assert kw_res.status_code == 429
    assert "quota" in kw_res.json()["detail"].lower()

    # Summarization blocked
    sum_res = await async_client.post(
        "/api/v1/nlp/summarization",
        headers=headers,
        json={"text": "Long text to summarize."},
    )
    assert sum_res.status_code == 429
    assert "quota" in sum_res.json()["detail"].lower()
