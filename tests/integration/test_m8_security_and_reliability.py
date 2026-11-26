"""M8 Security, integration boundary, failure, and reliability regression tests."""

import asyncio
import io
import logging
import time
import uuid
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from src.core.config import get_settings
from src.core.device import DeviceManager
from src.core.exceptions import ModelInferenceError, ModelLoadError, ModelResourceError
from src.core.logging import JSONFormatter
from src.core.models.base import BaseModel, ModelMetadata, ModelState
from src.core.models.nlp import SentimentModel, ZeroShotClassificationModel
from src.core.rate_limiter import rate_limiter
from src.db.base import Base
from src.db.session import get_db
from src.main import app, lifespan
from src.models.developer import Developer
from src.models.usage_record import UsageRecord
from src.services.api_key_service import APIKeyService
from src.services.developer_service import DeveloperService
from src.services.model_registry import ModelRegistry
from src.services.ollama_client import OllamaClient
from src.services.usage_service import quota_reservation_manager

settings = get_settings()


@pytest.fixture(autouse=True)
async def reset_state():
    await rate_limiter.reset()
    await quota_reservation_manager.reset()
    yield
    await rate_limiter.reset()
    await quota_reservation_manager.reset()


@pytest.fixture
async def dev_with_key(async_client: AsyncClient):
    res = await async_client.post(
        "/api/v1/developers",
        json={"email": f"m8_{uuid.uuid4().hex[:8]}@example.com", "name": "M8 Sec Dev"},
    )
    assert res.status_code == 201
    data = res.json()
    return {"X-API-Key": data["api_key"]}, data["id"], data["key_id"]


@pytest.mark.asyncio
async def test_m8_empty_scopes_cannot_escalate_to_analyze_read(async_client: AsyncClient, db_session: AsyncSession):
    """SECURITY: Passing scopes=[] must not bypass delegation guard or fall back to ['analyze:read']."""
    dev = await DeveloperService(db_session).create_developer(
        email="keys_write_only@example.com", name="Keys Write User"
    )
    _, keys_write_raw = await APIKeyService(db_session).create_key(
        developer_id=dev.id,
        name="Keys Write Only",
        scopes=["keys:write"],
    )

    # HTTP boundary must reject empty scopes with 422
    res = await async_client.post(
        "/api/v1/keys",
        headers={"X-API-Key": keys_write_raw},
        json={"name": "Escalated Key", "scopes": []},
    )
    assert res.status_code == 422

    # Service layer must also reject empty scopes explicitly rather than granting analyze:read
    with pytest.raises(ValueError):
        await APIKeyService(db_session).create_key(
            developer_id=dev.id,
            name="Service Empty Scopes",
            scopes=[],
        )


@pytest.mark.asyncio
async def test_m8_key_and_usage_endpoints_enforce_scopes(async_client: AsyncClient, dev_with_key: tuple):
    """SECURITY: Restricted sentiment-only key must be blocked (403) from reading usage or listing/revoking keys."""
    bootstrap_headers, _, bootstrap_key_id = dev_with_key

    # Mint restricted key with ONLY sentiment:read
    child_res = await async_client.post(
        "/api/v1/keys",
        headers=bootstrap_headers,
        json={"name": "Sentiment Worker", "scopes": ["sentiment:read"]},
    )
    assert child_res.status_code == 201
    restricted_headers = {"X-API-Key": child_res.json()["api_key"]}

    # Cannot read usage summary
    assert (await async_client.get("/api/v1/usage/summary", headers=restricted_headers)).status_code == 403
    # Cannot list keys
    assert (await async_client.get("/api/v1/keys", headers=restricted_headers)).status_code == 403
    # Cannot mint new keys
    assert (
        await async_client.post(
            "/api/v1/keys",
            headers=restricted_headers,
            json={"name": "Shadow Key", "scopes": ["sentiment:read"]},
        )
    ).status_code == 403
    # Cannot revoke developer's bootstrap key (DoS)
    assert (
        await async_client.delete(f"/api/v1/keys/{bootstrap_key_id}", headers=restricted_headers)
    ).status_code == 403

    # Verify keys:read can list keys but not create/delete
    read_key_res = await async_client.post(
        "/api/v1/keys",
        headers=bootstrap_headers,
        json={"name": "Key Reader", "scopes": ["keys:read"]},
    )
    read_headers = {"X-API-Key": read_key_res.json()["api_key"]}
    assert (await async_client.get("/api/v1/keys", headers=read_headers)).status_code == 200
    assert (
        await async_client.post(
            "/api/v1/keys",
            headers=read_headers,
            json={"name": "Denied", "scopes": ["keys:read"]},
        )
    ).status_code == 403


@pytest.mark.asyncio
async def test_m8_model_routes_support_slashed_huggingface_ids(
    async_client: AsyncClient, dev_with_key: tuple, db_session: AsyncSession
):
    """INTEGRATION: GET /models/{model_id} and POST /models/{model_id}/unload work for slashed HF IDs."""
    headers, dev_id, _ = dev_with_key
    reg = ModelRegistry(device_mgr=DeviceManager("cpu"))
    meta = ModelMetadata(model_id="dslim/bert-base-NER", name="Slashed NER", task="ner")
    reg.register(meta, lambda: SentimentModel(meta, pipeline_factory=lambda: lambda x: [{"label": "POSITIVE", "score": 0.9}]))
    await reg.get_model("dslim/bert-base-NER")
    app.state.model_registry = reg

    # GET status with slashed model_id
    res_get = await async_client.get("/api/v1/models/dslim/bert-base-NER", headers=headers)
    assert res_get.status_code == 200
    assert res_get.json()["model_id"] == "dslim/bert-base-NER"

    # Admin key can unload slashed model_id
    _, admin_raw = await APIKeyService(db_session).create_key(
        developer_id=uuid.UUID(dev_id), name="Admin", scopes=["admin"]
    )
    res_unload = await async_client.post(
        "/api/v1/models/dslim/bert-base-NER/unload",
        headers={"X-API-Key": admin_raw},
    )
    assert res_unload.status_code == 200
    assert res_unload.json()["model_id"] == "dslim/bert-base-NER"
    assert not reg.is_loaded("dslim/bert-base-NER")


@pytest.mark.asyncio
async def test_m8_concurrent_single_requests_cannot_race_past_quota():
    """CONCURRENCY: Concurrent single-item /nlp/sentiment requests cannot exceed remaining quota."""
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

    reg = ModelRegistry(device_mgr=DeviceManager("cpu"))

    def slow_sent(text: str):
        time.sleep(0.06)
        return [{"label": "POSITIVE", "score": 0.99}, {"label": "NEGATIVE", "score": 0.01}]

    meta = ModelMetadata(model_id=settings.MODEL_SENTIMENT_ID, name="SlowS", task="sentiment")
    reg.register(meta, lambda: SentimentModel(meta, pipeline_factory=lambda: slow_sent))
    app.state.model_registry = reg
    app.dependency_overrides[get_db] = override_get_db

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            reg_res = await client.post(
                "/api/v1/developers",
                json={"email": "single_race@example.com", "name": "Single Race"},
            )
            dev_id = uuid.UUID(reg_res.json()["id"])
            headers = {"X-API-Key": reg_res.json()["api_key"]}

            async with session_factory() as db:
                dev = await db.get(Developer, dev_id)
                dev.monthly_request_quota = 1
                await db.commit()

            r1, r2 = await asyncio.gather(
                client.post("/api/v1/nlp/sentiment", headers=headers, json={"text": "Great one!"}),
                client.post("/api/v1/nlp/sentiment", headers=headers, json={"text": "Great two!"}),
            )
            assert sorted([r1.status_code, r2.status_code]) == [200, 429]

            async with session_factory() as db:
                rows = (await db.execute(select(UsageRecord).where(UsageRecord.developer_id == dev_id))).scalars().all()
                assert len(rows) == 1
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()


@pytest.mark.asyncio
async def test_m8_database_failure_returns_safe_503_with_security_headers():
    """RELIABILITY: Database errors return safe JSON 503 without leaking connection details and keep security headers."""
    async def broken_db():
        mock_sess = AsyncMock(spec=AsyncSession)
        mock_sess.execute.side_effect = OperationalError("SELECT 1", {}, Exception("secret_pg_password_or_host"))
        yield mock_sess

    app.dependency_overrides[get_db] = broken_db
    try:
        async with AsyncClient(transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test") as c:
            res = await c.get("/api/v1/developers/me", headers={"X-API-Key": "nlp_live_abc"})
            assert res.status_code == 503
            body = res.json()
            assert body["detail"] == "Database service temporarily unavailable."
            assert "secret_pg_password" not in res.text
            assert res.headers.get("X-Content-Type-Options") == "nosniff"
            assert res.headers.get("X-Frame-Options") == "DENY"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_m8_validation_hardening_zeroshot_dos_blank_names_and_ollama_truncation(
    async_client: AsyncClient, dev_with_key: tuple
):
    """SECURITY: Bound zero-shot candidate_labels, reject blank names, and truncate huge Ollama error responses."""
    headers, _ , _ = dev_with_key

    # Blank developer name rejected
    blank_dev = await async_client.post(
        "/api/v1/developers",
        json={"email": "blank_name@example.com", "name": "   "},
    )
    assert blank_dev.status_code == 422

    # Blank API key name rejected
    blank_key = await async_client.post(
        "/api/v1/keys",
        headers=headers,
        json={"name": "   ", "scopes": ["sentiment:read"]},
    )
    assert blank_key.status_code == 422

    # Combinatorial zero-shot NLI label explosion blocked (> 25 labels)
    reg = ModelRegistry(device_mgr=DeviceManager("cpu"))
    z_meta = ModelMetadata(model_id=settings.MODEL_ZERO_SHOT_ID, name="ZS", task="zero-shot")
    reg.register(
        z_meta,
        lambda: ZeroShotClassificationModel(
            z_meta, pipeline_factory=lambda: lambda t, candidate_labels, multi_label: {"labels": candidate_labels, "scores": [0.5] * len(candidate_labels)}
        ),
    )
    app.state.model_registry = reg

    too_many_labels = [f"label_{i}" for i in range(30)]
    zs_res = await async_client.post(
        "/api/v1/nlp/classification",
        headers=headers,
        json={"text": "Classify this document.", "candidate_labels": too_many_labels},
    )
    assert zs_res.status_code == 422

    # Ollama huge error body truncated safely
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "INTERNAL_ERR_" + ("X" * 2000)
    mock_http.post = AsyncMock(return_value=mock_resp)
    client = OllamaClient(http_client=mock_http)
    with pytest.raises(Exception) as exc_info:
        await client.generate("test")
    assert len(str(exc_info.value)) < 500


@pytest.mark.asyncio
async def test_m8_model_load_failure_and_lifespan_shutdown_safety(async_client: AsyncClient, dev_with_key: tuple, db_session):
    """RELIABILITY: ModelLoadError charges 0 quota, recovers on retry, and lifespan shutdown handles busy models."""
    headers, dev_id, _ = dev_with_key

    should_fail = True

    class FlakyModel(BaseModel):
        def _load_weights(self, device: str) -> None:
            if should_fail:
                raise RuntimeError("Corrupted weight file /internal/path/weights.bin")

        def _unload_weights(self) -> None:
            pass

        def _execute_inference(self, inputs, **kwargs):
            return {"label": "POSITIVE", "confidence": 0.99, "scores": {"POSITIVE": 0.99, "NEGATIVE": 0.01}}

    reg = ModelRegistry(device_mgr=DeviceManager("cpu"))
    meta = ModelMetadata(model_id=settings.MODEL_SENTIMENT_ID, name="Flaky", task="sentiment")
    reg.register(meta, lambda: FlakyModel(meta))
    app.state.model_registry = reg

    fail_res = await async_client.post(
        "/api/v1/nlp/sentiment",
        headers=headers,
        json={"text": "Should fail on model load"},
    )
    assert fail_res.status_code == 500
    assert quota_reservation_manager.get_reserved(uuid.UUID(dev_id)) == 0
    rows = (await db_session.execute(select(UsageRecord).where(UsageRecord.developer_id == uuid.UUID(dev_id)))).scalars().all()
    assert len(rows) == 0

    # Recover cleanly on retry once weights are available
    should_fail = False
    ok_res = await async_client.post(
        "/api/v1/nlp/sentiment",
        headers=headers,
        json={"text": "Works after recovery"},
    )
    assert ok_res.status_code == 200

    # Lifespan startup + shutdown with a busy model must not crash
    async with lifespan(app):
        busy_meta = ModelMetadata(model_id="busy-shutdown-model", name="Busy", task="test")
        busy_model = FlakyModel(busy_meta)
        app.state.model_registry.register(busy_meta, lambda: busy_model)
        await app.state.model_registry.get_model("busy-shutdown-model")
        busy_model._active_inferences = 1
        # Exit lifespan -> unload_all catches ModelInUseError cleanly
    busy_model._active_inferences = 0


@pytest.mark.asyncio
async def test_m8_logs_never_contain_plaintext_api_key(async_client: AsyncClient):
    """SECURITY: Structured JSON logs during registration/auth must not emit plaintext API keys."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger("nlpulse")
    root.addHandler(handler)
    try:
        reg_res = await async_client.post(
            "/api/v1/developers",
            json={"email": "log_safety@example.com", "name": "Log Safety"},
        )
        raw_key = reg_res.json()["api_key"]
        await async_client.get("/api/v1/developers/me", headers={"X-API-Key": raw_key})
        log_output = stream.getvalue()
        assert raw_key not in log_output
    finally:
        root.removeHandler(handler)
