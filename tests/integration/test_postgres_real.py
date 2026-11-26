"""Opt-in integration tests executed against a real PostgreSQL 16 database instance.

Set NLPULSE_TEST_POSTGRES_URL (e.g. postgresql+asyncpg://user:pw@localhost:5432/db) to run.
"""

import asyncio
import os
import time
import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import get_settings
from src.core.device import DeviceManager
from src.core.models.base import ModelMetadata
from src.core.models.nlp import SentimentModel
from src.core.rate_limiter import rate_limiter
from src.db.session import get_db
from src.main import app
from src.models.api_key import APIKey
from src.models.developer import Developer
from src.models.usage_record import UsageRecord
from src.services.api_key_service import APIKeyService
from src.services.developer_service import DeveloperService
from src.services.model_registry import ModelRegistry
from src.services.usage_service import quota_reservation_manager

settings = get_settings()
PG_URL = os.getenv("NLPULSE_TEST_POSTGRES_URL")

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        not PG_URL,
        reason="Skipping real PostgreSQL tests: NLPULSE_TEST_POSTGRES_URL not set",
    ),
]


@pytest.mark.asyncio
async def test_postgres_orm_constraints_rollbacks_and_cascades():
    """Verify real PostgreSQL 16 ORM CRUD, rollbacks, check constraints, and foreign key cascades."""
    assert PG_URL is not None
    engine = create_async_engine(PG_URL, echo=False)
    sf = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)

    dev_id = None
    try:
        async with sf() as db:
            ver = (await db.execute(text("SELECT version()"))).scalar_one()
            assert "PostgreSQL 16" in ver

            dev = await DeveloperService(db).create_developer(
                email=f"pg_crud_{uuid.uuid4().hex[:8]}@example.com",
                name="PG CRUD Dev",
            )
            dev_id = dev.id
            key, _ = await APIKeyService(db).create_key(developer_id=dev.id, name="PG Key", scopes=["sentiment:read"])
            rec = UsageRecord(
                developer_id=dev.id,
                api_key_id=key.id,
                endpoint="/api/v1/nlp/sentiment",
                method="POST",
                status_code=200,
                input_character_count=15,
                latency_ms=5.0,
                cost_units=1,
            )
            db.add(rec)
            await db.commit()

            # Check constraint enforced by PostgreSQL
            bad_rec = UsageRecord(
                developer_id=dev.id,
                endpoint="/api/v1/nlp/sentiment",
                method="POST",
                status_code=999,
                input_character_count=10,
                latency_ms=1.0,
            )
            db.add(bad_rec)
            with pytest.raises(IntegrityError):
                await db.commit()
            await db.rollback()

            # Deleting API key preserves UsageRecord with api_key_id = NULL
            await db.delete(key)
            await db.commit()
            await db.refresh(rec)
            assert rec.api_key_id is None

            # Deleting Developer cascades to UsageRecord
            saved_dev_id = dev_id
            loaded_dev = await db.get(Developer, saved_dev_id)
            await db.delete(loaded_dev)
            await db.commit()
            dev_id = None

            leftover = (await db.execute(select(UsageRecord).where(UsageRecord.developer_id == saved_dev_id))).scalars().all()
            assert leftover == []
    finally:
        if dev_id is not None:
            async with sf() as cleanup:
                d = await cleanup.get(Developer, dev_id)
                if d:
                    await cleanup.delete(d)
                    await cleanup.commit()
        await engine.dispose()


@pytest.mark.asyncio
async def test_postgres_concurrent_batch_and_single_requests_no_deadlock_or_quota_breach():
    """Verify real PostgreSQL 16 concurrent HTTP batch + single requests do not deadlock or exceed quota."""
    assert PG_URL is not None
    await rate_limiter.reset()
    await quota_reservation_manager.reset()

    engine = create_async_engine(PG_URL, pool_size=10, max_overflow=10)
    sf = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)

    async def override_db():
        async with sf() as session:
            yield session

    reg = ModelRegistry(device_mgr=DeviceManager("cpu"))

    def slow_sentiment(text: str):
        time.sleep(0.08)
        return [{"label": "POSITIVE", "score": 0.99}, {"label": "NEGATIVE", "score": 0.01}]

    meta = ModelMetadata(model_id=settings.MODEL_SENTIMENT_ID, name="PG Slow", task="sentiment")
    reg.register(meta, lambda: SentimentModel(meta, pipeline_factory=lambda: slow_sentiment))
    app.state.model_registry = reg
    app.dependency_overrides[get_db] = override_db

    dev_id = None
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            reg_res = await client.post(
                "/api/v1/developers",
                json={"email": f"pg_conc_{uuid.uuid4().hex[:8]}@example.com", "name": "PG Concurrent"},
            )
            assert reg_res.status_code == 201
            dev_id = uuid.UUID(reg_res.json()["id"])
            headers = {"X-API-Key": reg_res.json()["api_key"]}

            async with sf() as db:
                dev = await db.get(Developer, dev_id)
                dev.monthly_request_quota = 10
                await db.commit()

            # Two concurrent batches of 6 against quota=10 (6+6=12 > 10): must finish fast (no PG deadlock)
            batch_payload = {"items": [{"text": f"Great item {i}"} for i in range(6)]}
            res_a, res_b = await asyncio.wait_for(
                asyncio.gather(
                    client.post("/api/v1/nlp/batch/sentiment", headers=headers, json=batch_payload),
                    client.post("/api/v1/nlp/batch/sentiment", headers=headers, json=batch_payload),
                ),
                timeout=5.0,
            )
            assert sorted([res_a.status_code, res_b.status_code]) == [200, 429]

            # Remaining quota is now 4; fire 6 concurrent single requests (only 4 may pass)
            single_responses = await asyncio.wait_for(
                asyncio.gather(
                    *[
                        client.post("/api/v1/nlp/sentiment", headers=headers, json={"text": f"Great {i}"})
                        for i in range(6)
                    ]
                ),
                timeout=5.0,
            )
            statuses = [r.status_code for r in single_responses]
            assert statuses.count(200) == 4
            assert statuses.count(429) == 2

            async with sf() as db:
                rows = (await db.execute(select(UsageRecord).where(UsageRecord.developer_id == dev_id))).scalars().all()
                assert len(rows) == 10
    finally:
        app.dependency_overrides.clear()
        if dev_id is not None:
            async with sf() as cleanup:
                d = await cleanup.get(Developer, dev_id)
                if d:
                    await cleanup.delete(d)
                    await cleanup.commit()
        await engine.dispose()
        await rate_limiter.reset()
        await quota_reservation_manager.reset()
