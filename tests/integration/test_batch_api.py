"""Integration tests for M7 Batch Processing endpoints across all 6 NLP capabilities."""

import asyncio
import time
import uuid
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from src.api import deps
from src.core.config import get_settings
from src.core.device import DeviceManager
from src.core.exceptions import OllamaTimeoutError
from src.core.models.base import ModelMetadata
from src.core.models.nlp import (
    EmbeddingModel,
    NERModel,
    SentimentModel,
    SummarizationModel,
    ZeroShotClassificationModel,
)
from src.core.rate_limiter import SlidingWindowRateLimiter, rate_limiter
from src.db.base import Base
from src.db.session import get_db
from src.main import app
from src.models.developer import Developer
from src.models.usage_record import UsageRecord
from src.services.model_registry import ModelRegistry
from src.services.ollama_client import OllamaClient
from src.services.usage_service import quota_reservation_manager

settings = get_settings()


def _build_test_registry_and_ollama():
    reg = ModelRegistry(device_mgr=DeviceManager(preference="cpu"), max_cached_models=5)

    # 1. Sentiment
    class FakeHFSentiment:
        tokenizer = object()
        model = object()

        def __call__(self, inputs, **kwargs):
            if isinstance(inputs, list):
                return [self(t) for t in inputs]
            if "great" in inputs.lower():
                return [{"label": "POSITIVE", "score": 0.99}, {"label": "NEGATIVE", "score": 0.01}]
            return [{"label": "NEGATIVE", "score": 0.96}, {"label": "POSITIVE", "score": 0.04}]

    s_meta = ModelMetadata(model_id=settings.MODEL_SENTIMENT_ID, name="Sent", task="sentiment")
    reg.register(s_meta, lambda: SentimentModel(s_meta, pipeline_factory=lambda: FakeHFSentiment()))

    # 2. NER
    class FakeHFNER:
        tokenizer = object()
        model = object()

        def __call__(self, inputs, **kwargs):
            if isinstance(inputs, list):
                return [self(t) for t in inputs]
            if "sundar" in inputs.lower():
                return [{"entity_group": "PER", "score": 0.99, "word": "Sundar Pichai", "start": 0, "end": 13}]
            return []

    n_meta = ModelMetadata(model_id=settings.MODEL_NER_ID, name="NER", task="ner")
    reg.register(n_meta, lambda: NERModel(n_meta, pipeline_factory=lambda: FakeHFNER()))

    # 3. Zero-Shot
    class FakeHFZeroShot:
        tokenizer = object()
        model = object()

        def __call__(self, inputs, candidate_labels, multi_label=False, **kwargs):
            if isinstance(inputs, list):
                return [self(t, candidate_labels, multi_label) for t in inputs]
            return {
                "sequence": inputs,
                "labels": list(candidate_labels),
                "scores": [round(0.8 / (i + 1), 4) for i in range(len(candidate_labels))],
            }

    z_meta = ModelMetadata(model_id=settings.MODEL_ZERO_SHOT_ID, name="ZS", task="zero-shot")
    reg.register(z_meta, lambda: ZeroShotClassificationModel(z_meta, pipeline_factory=lambda: FakeHFZeroShot()))

    # 4. Embedding
    def fake_emb(texts: list):
        vecs = []
        for t in texts:
            if "fastapi" in t.lower() or "api" in t.lower():
                v = np.array([0.9, 0.1, 0.0], dtype=np.float32)
            else:
                v = np.array([0.3, 0.3, 0.3], dtype=np.float32)
            vecs.append(v / np.linalg.norm(v))
        return np.array(vecs, dtype=np.float32)

    e_meta = ModelMetadata(model_id=settings.MODEL_EMBEDDING_ID, name="Emb", task="embedding")
    reg.register(e_meta, lambda: EmbeddingModel(e_meta, pipeline_factory=lambda: fake_emb))

    # 5. Summarization
    def fake_sum(text: str, **kwargs):
        return [{"summary_text": f"Summary: {text[:30]}..."}]

    sum_meta = ModelMetadata(model_id=settings.MODEL_SUMMARIZATION_ID, name="Sum", task="summarization")
    reg.register(sum_meta, lambda: SummarizationModel(sum_meta, pipeline_factory=lambda: fake_sum))

    # 6. Ollama mock
    async def fake_generate(prompt: str, **kwargs):
        if "TIMEOUT_ITEM" in prompt:
            raise OllamaTimeoutError("Timed out")
        return {
            "response": '{"company": "Google LLC", "ceo": "Sundar Pichai"}',
            "prompt_tokens": 100,
            "completion_tokens": 100,
            "total_tokens": 200,
            "total_duration_ns": 150000000,
            "model": "qwen2.5:3b",
        }

    ollama_client = OllamaClient()
    ollama_client.generate = fake_generate  # type: ignore[method-assign]
    return reg, ollama_client


@pytest.fixture(autouse=True)
async def setup_batch_mocks():
    await rate_limiter.reset()
    await quota_reservation_manager.reset()
    reg, ollama_client = _build_test_registry_and_ollama()
    app.state.model_registry = reg
    app.state.ollama_client = ollama_client
    yield
    await rate_limiter.reset()
    await quota_reservation_manager.reset()


@pytest.fixture
async def dev_auth(async_client: AsyncClient):
    res = await async_client.post(
        "/api/v1/developers",
        json={"email": "m7_batch_dev@example.com", "name": "M7 Batch Dev"},
    )
    assert res.status_code == 201
    data = res.json()
    return {"X-API-Key": data["api_key"]}, data["id"]


@pytest.mark.asyncio
async def test_batch_all_six_endpoints_success_and_metering(async_client: AsyncClient, dev_auth: tuple, db_session):
    """Verify all six batch endpoints succeed, preserve order/IDs, and create per-item UsageRecords."""
    headers, dev_id = dev_auth

    # 1. Sentiment
    r1 = await async_client.post(
        "/api/v1/nlp/batch/sentiment",
        headers=headers,
        json={"items": [{"id": "s1", "text": "Great API!"}, {"id": "s2", "text": "Bad bug."}]},
    )
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["capability"] == "sentiment"
    assert d1["succeeded_count"] == 2
    assert d1["results"][0]["id"] == "s1"
    assert d1["results"][0]["result"]["label"] == "POSITIVE"
    assert d1["results"][1]["result"]["label"] == "NEGATIVE"

    # 2. NER
    r2 = await async_client.post(
        "/api/v1/nlp/batch/ner",
        headers=headers,
        json={"items": [{"id": "n1", "text": "Sundar Pichai leads Google."}]},
    )
    assert r2.status_code == 200
    assert r2.json()["results"][0]["result"]["entity_count"] == 1

    # 3. Zero-Shot Classification
    r3 = await async_client.post(
        "/api/v1/nlp/batch/classification",
        headers=headers,
        json={
            "candidate_labels": ["technology", "finance"],
            "items": [{"id": "c1", "text": "New Python framework released."}],
        },
    )
    assert r3.status_code == 200
    assert r3.json()["results"][0]["result"]["top_label"] == "technology"

    # 4. Keywords
    r4 = await async_client.post(
        "/api/v1/nlp/batch/keywords",
        headers=headers,
        json={
            "max_keywords": 2,
            "items": [{"id": "k1", "text": "FastAPI builds fast web APIs."}],
        },
    )
    assert r4.status_code == 200
    assert len(r4.json()["results"][0]["result"]["keywords"]) == 2

    # 5. Summarization
    r5 = await async_client.post(
        "/api/v1/nlp/batch/summarization",
        headers=headers,
        json={
            "min_length": 10,
            "max_length": 50,
            "items": [{"id": "sum1", "text": "PostgreSQL is a reliable relational database system."}],
        },
    )
    assert r5.status_code == 200
    assert "Summary:" in r5.json()["results"][0]["result"]["summary"]

    # 6. Extraction
    r6 = await async_client.post(
        "/api/v1/nlp/batch/extraction",
        headers=headers,
        json={
            "fields": [{"name": "company", "type": "string"}, {"name": "ceo", "type": "string"}],
            "items": [{"id": "ext1", "text": "Google LLC is led by Sundar Pichai."}],
        },
    )
    assert r6.status_code == 200
    assert r6.json()["results"][0]["result"]["extracted_data"]["company"] == "Google LLC"

    rows = (
        await db_session.execute(select(UsageRecord).where(UsageRecord.developer_id == uuid.UUID(dev_id)))
    ).scalars().all()
    assert len(rows) == 7  # 2 + 1 + 1 + 1 + 1 + 1


@pytest.mark.asyncio
async def test_batch_empty_and_oversized_rejected_422(async_client: AsyncClient, dev_auth: tuple, monkeypatch):
    """Verify empty batches and batches exceeding BATCH_MAX_SIZE return HTTP 422."""
    headers, _ = dev_auth
    monkeypatch.setattr(settings, "BATCH_MAX_SIZE", 2)

    empty_res = await async_client.post("/api/v1/nlp/batch/sentiment", headers=headers, json={"items": []})
    assert empty_res.status_code == 422

    over_res = await async_client.post(
        "/api/v1/nlp/batch/sentiment",
        headers=headers,
        json={"items": [{"text": "a"}, {"text": "b"}, {"text": "c"}]},
    )
    assert over_res.status_code == 422


@pytest.mark.asyncio
async def test_batch_partial_failures_and_order_preservation(async_client: AsyncClient, dev_auth: tuple, db_session):
    """Verify mixed valid/invalid/failing items return 200 with safe per-item errors and charge only successes."""
    headers, dev_id = dev_auth

    res = await async_client.post(
        "/api/v1/nlp/batch/extraction",
        headers=headers,
        json={
            "fields": [{"name": "company", "type": "string"}],
            "items": [
                {"id": "first", "text": "Google LLC launched a model."},
                {"id": "second-empty", "text": "   "},
                {"id": "third-timeout", "text": "TIMEOUT_ITEM document"},
                {"id": "fourth", "text": "Sundar Pichai spoke at keynote."},
            ],
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["total_items"] == 4
    assert body["succeeded_count"] == 2
    assert body["failed_count"] == 2
    assert [r["id"] for r in body["results"]] == ["first", "second-empty", "third-timeout", "fourth"]
    assert body["results"][0]["status"] == "success"
    assert body["results"][1]["error"]["status_code"] == 422
    assert body["results"][2]["error"]["status_code"] == 504
    assert body["results"][3]["status"] == "success"

    rows = (
        await db_session.execute(select(UsageRecord).where(UsageRecord.developer_id == uuid.UUID(dev_id)))
    ).scalars().all()
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_batch_auth_granular_scopes_and_analyze_read_compat(async_client: AsyncClient, dev_auth: tuple):
    """Verify 401 unauthenticated, 403 granular scope rejection, and 200 legacy analyze:read access."""
    headers, _ = dev_auth

    # Unauthenticated -> 401
    unauth = await async_client.post("/api/v1/nlp/batch/sentiment", json={"items": [{"text": "hello"}]})
    assert unauth.status_code == 401

    # Granular key: sentiment:read allowed on /batch/sentiment, forbidden on /batch/ner
    s_key = (
        await async_client.post(
            "/api/v1/keys",
            headers=headers,
            json={"name": "Sentiment Only", "scopes": ["sentiment:read"]},
        )
    ).json()["api_key"]

    assert (
        await async_client.post(
            "/api/v1/nlp/batch/sentiment",
            headers={"X-API-Key": s_key},
            json={"items": [{"text": "Great"}]},
        )
    ).status_code == 200
    assert (
        await async_client.post(
            "/api/v1/nlp/batch/ner",
            headers={"X-API-Key": s_key},
            json={"items": [{"text": "Alice"}]},
        )
    ).status_code == 403
    assert (
        await async_client.post(
            "/api/v1/nlp/batch/classification",
            headers={"X-API-Key": s_key},
            json={"candidate_labels": ["a"], "items": [{"text": "Alice"}]},
        )
    ).status_code == 403
    assert (
        await async_client.post(
            "/api/v1/nlp/batch/keywords",
            headers={"X-API-Key": s_key},
            json={"items": [{"text": "FastAPI"}]},
        )
    ).status_code == 403
    assert (
        await async_client.post(
            "/api/v1/nlp/batch/summarization",
            headers={"X-API-Key": s_key},
            json={"items": [{"text": "FastAPI"}]},
        )
    ).status_code == 403
    assert (
        await async_client.post(
            "/api/v1/nlp/batch/extraction",
            headers={"X-API-Key": s_key},
            json={"fields": [{"name": "company"}], "items": [{"text": "Google"}]},
        )
    ).status_code == 403

    # Legacy analyze:read key works on all 6 batch endpoints
    legacy_key = (
        await async_client.post(
            "/api/v1/keys",
            headers=headers,
            json={"name": "Legacy Key", "scopes": ["analyze:read"]},
        )
    ).json()["api_key"]
    legacy_headers = {"X-API-Key": legacy_key}

    assert (
        await async_client.post(
            "/api/v1/nlp/batch/sentiment",
            headers=legacy_headers,
            json={"items": [{"text": "Great"}]},
        )
    ).status_code == 200
    assert (
        await async_client.post(
            "/api/v1/nlp/batch/ner",
            headers=legacy_headers,
            json={"items": [{"text": "Sundar"}]},
        )
    ).status_code == 200
    assert (
        await async_client.post(
            "/api/v1/nlp/batch/classification",
            headers=legacy_headers,
            json={"candidate_labels": ["tech"], "items": [{"text": "FastAPI"}]},
        )
    ).status_code == 200
    assert (
        await async_client.post(
            "/api/v1/nlp/batch/keywords",
            headers=legacy_headers,
            json={"items": [{"text": "FastAPI"}]},
        )
    ).status_code == 200
    assert (
        await async_client.post(
            "/api/v1/nlp/batch/summarization",
            headers=legacy_headers,
            json={"min_length": 10, "max_length": 40, "items": [{"text": "PostgreSQL database."}]},
        )
    ).status_code == 200
    assert (
        await async_client.post(
            "/api/v1/nlp/batch/extraction",
            headers=legacy_headers,
            json={"fields": [{"name": "company"}], "items": [{"text": "Google LLC"}]},
        )
    ).status_code == 200


@pytest.mark.asyncio
async def test_batch_rate_limiting_and_quota_exhaustion(async_client: AsyncClient, dev_auth: tuple, db_session):
    """Verify batch endpoints enforce process-local rate limit (429) and monthly quota (429)."""
    headers, dev_id = dev_auth

    test_limiter = SlidingWindowRateLimiter(default_max_requests=1, default_window_seconds=30)
    orig_limiter = deps.rate_limiter
    deps.rate_limiter = test_limiter
    try:
        ok_res = await async_client.post(
            "/api/v1/nlp/batch/sentiment",
            headers=headers,
            json={"items": [{"text": "Great!"}]},
        )
        assert ok_res.status_code == 200

        rl_res = await async_client.post(
            "/api/v1/nlp/batch/sentiment",
            headers=headers,
            json={"items": [{"text": "Great!"}]},
        )
        assert rl_res.status_code == 429
        assert "retry-after" in rl_res.headers
    finally:
        deps.rate_limiter = orig_limiter

    # Quota exhaustion -> 429
    dev = await db_session.get(Developer, uuid.UUID(dev_id))
    assert dev is not None
    dev.monthly_request_quota = 1  # Already 1 UsageRecord committed above
    await db_session.commit()

    q_res = await async_client.post(
        "/api/v1/nlp/batch/sentiment",
        headers=headers,
        json={"items": [{"text": "Blocked by quota"}]},
    )
    assert q_res.status_code == 429
    assert "quota" in q_res.json()["detail"].lower()


@pytest.mark.asyncio
async def test_concurrent_http_batches_cannot_exceed_quota():
    """Verify two concurrent HTTP batch requests cannot exceed monthly quota through race conditions."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)

    async def override_get_db():
        async with session_factory() as session:
            yield session

    # Slow sentiment pipeline so both HTTP requests overlap during inference
    reg = ModelRegistry(device_mgr=DeviceManager(preference="cpu"))

    def slow_sentiment(text: str):
        time.sleep(0.05)
        return [{"label": "POSITIVE", "score": 0.99}, {"label": "NEGATIVE", "score": 0.01}]

    s_meta = ModelMetadata(model_id=settings.MODEL_SENTIMENT_ID, name="SlowSent", task="sentiment")
    reg.register(s_meta, lambda: SentimentModel(s_meta, pipeline_factory=lambda: slow_sentiment))

    app.state.model_registry = reg
    app.dependency_overrides[get_db] = override_get_db

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            reg_res = await client.post(
                "/api/v1/developers",
                json={"email": "concurrent_http_batch@example.com", "name": "Race HTTP"},
            )
            assert reg_res.status_code == 201
            dev_id = uuid.UUID(reg_res.json()["id"])
            headers = {"X-API-Key": reg_res.json()["api_key"]}

            async with session_factory() as db:
                dev = await db.get(Developer, dev_id)
                assert dev is not None
                dev.monthly_request_quota = 10
                await db.commit()

            payload = {"items": [{"text": f"Great item {i}"} for i in range(6)]}
            res_a, res_b = await asyncio.gather(
                client.post("/api/v1/nlp/batch/sentiment", headers=headers, json=payload),
                client.post("/api/v1/nlp/batch/sentiment", headers=headers, json=payload),
            )
            statuses = sorted([res_a.status_code, res_b.status_code])
            assert statuses == [200, 429]

            async with session_factory() as db:
                rows = (await db.execute(select(UsageRecord).where(UsageRecord.developer_id == dev_id))).scalars().all()
                assert len(rows) == 6
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()
