"""Unit tests for M7 atomic batch quota reservation, settlement, and concurrency safety."""

import asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
import pytest

from src.core.exceptions import QuotaExceededError
from src.db.base import Base
from src.models.usage_record import UsageRecord
from src.services.developer_service import DeveloperService
from src.services.usage_service import UsageService, quota_reservation_manager


@pytest.fixture(autouse=True)
async def reset_reservations():
    """Ensure reservation ledger starts clean for every test."""
    await quota_reservation_manager.reset()
    yield
    await quota_reservation_manager.reset()


@pytest.mark.asyncio
async def test_reserve_and_settle_partial_batch_charges_only_succeeded(db_session: AsyncSession):
    """Verify reserving 5 units and settling 3 succeeded items commits only 3 records and frees all 5."""
    dev = await DeveloperService(db_session).create_developer(
        email="partial_quota@example.com",
        name="Partial Quota",
    )
    dev.monthly_request_quota = 5
    await db_session.commit()

    usage_svc = UsageService(db_session)
    await usage_svc.reserve_quota(dev, units=5)
    assert quota_reservation_manager.get_reserved(dev.id) == 5

    # Cannot reserve another unit while 5 are in flight
    with pytest.raises(QuotaExceededError):
        await usage_svc.reserve_quota(dev, units=1)

    records = [
        UsageRecord(
            developer_id=dev.id,
            endpoint="/api/v1/nlp/batch/sentiment",
            method="POST",
            status_code=200,
            input_character_count=10,
            latency_ms=5.0,
            cost_units=1,
        )
        for _ in range(3)
    ]
    await usage_svc.settle_batch_usage(dev.id, reserved_units=5, usage_records=records)

    assert quota_reservation_manager.get_reserved(dev.id) == 0
    assert await usage_svc.get_monthly_usage_count(dev.id) == 3

    # Remaining 2 units are now available
    await usage_svc.reserve_quota(dev, units=2)
    await usage_svc.release_reservation(dev.id, units=2)
    assert quota_reservation_manager.get_reserved(dev.id) == 0


@pytest.mark.asyncio
async def test_concurrent_quota_reservations_prevent_oversubscription():
    """Verify concurrent batches cannot race past monthly_request_quota."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as setup_session:
        dev = await DeveloperService(setup_session).create_developer(
            email="race_quota@example.com",
            name="Race Guard",
        )
        dev.monthly_request_quota = 10
        await setup_session.commit()
        dev_id = dev.id

    async def run_batch_attempt(item_count: int) -> str:
        async with session_factory() as session:
            svc = UsageService(session)
            loaded_dev = await DeveloperService(session).get_required_developer(dev_id)
            try:
                await svc.reserve_quota(loaded_dev, units=item_count)
            except QuotaExceededError:
                return "rejected"
            try:
                # Simulate slow model inference while holding reservation
                await asyncio.sleep(0.05)
                recs = [
                    UsageRecord(
                        developer_id=dev_id,
                        endpoint="/api/v1/nlp/batch/sentiment",
                        method="POST",
                        status_code=200,
                        input_character_count=12,
                        latency_ms=2.0,
                        cost_units=1,
                    )
                    for _ in range(item_count)
                ]
                await svc.settle_batch_usage(dev_id, reserved_units=item_count, usage_records=recs)
                return "accepted"
            except Exception:
                await svc.release_reservation(dev_id, units=item_count)
                raise

    # Two concurrent batches of 6 against quota=10 (6+6=12 > 10): exactly 1 must accept, 1 must reject
    outcomes = await asyncio.gather(run_batch_attempt(6), run_batch_attempt(6))
    assert sorted(outcomes) == ["accepted", "rejected"]
    assert quota_reservation_manager.get_reserved(dev_id) == 0

    async with session_factory() as check_session:
        rows = (await check_session.execute(select(UsageRecord).where(UsageRecord.developer_id == dev_id))).scalars().all()
        assert len(rows) == 6

    await engine.dispose()
