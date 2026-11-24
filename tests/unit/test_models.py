"""Unit tests for database models and relationships."""

from datetime import datetime, timedelta, timezone
import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.api_key import APIKey
from src.models.developer import Developer
from src.models.usage_record import UsageRecord


@pytest.mark.asyncio
async def test_create_developer(db_session: AsyncSession):
    """Verify Developer model creation with default values and timestamps."""
    dev = Developer(
        email="alice@example.com",
        name="Alice Developer",
        tier="free",
    )
    db_session.add(dev)
    await db_session.commit()
    await db_session.refresh(dev)

    assert dev.id is not None
    assert dev.email == "alice@example.com"
    assert dev.name == "Alice Developer"
    assert dev.tier == "free"
    assert dev.monthly_request_quota == 1000
    assert dev.is_active is True
    assert dev.created_at is not None
    assert dev.updated_at is not None


@pytest.mark.asyncio
async def test_developer_unique_email(db_session: AsyncSession):
    """Verify unique constraint on developer email."""
    dev1 = Developer(email="dup@example.com", name="Dev One")
    db_session.add(dev1)
    await db_session.commit()

    dev2 = Developer(email="dup@example.com", name="Dev Two")
    db_session.add(dev2)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_create_api_key_and_relationship(db_session: AsyncSession):
    """Verify APIKey creation, hash storage, and developer relationship."""
    dev = Developer(email="bob@example.com", name="Bob Dev")
    db_session.add(dev)
    await db_session.commit()
    await db_session.refresh(dev)

    key = APIKey(
        developer_id=dev.id,
        name="Test Key",
        key_hash="a" * 64,
        key_prefix="nlp_live_test",
        scopes=["sentiment:read", "ner:read"],
    )
    db_session.add(key)
    await db_session.commit()
    await db_session.refresh(key)

    assert key.id is not None
    assert key.developer_id == dev.id
    assert key.is_active is True
    assert key.scopes == ["sentiment:read", "ner:read"]
    assert key.revoked_at is None
    assert key.last_used_at is None

    # Expire cached developer or refresh to populate relationship
    await db_session.refresh(dev)
    stmt = select(Developer).where(Developer.id == dev.id)
    res = await db_session.execute(stmt)
    loaded_dev = res.scalar_one()
    assert len(loaded_dev.api_keys) == 1
    assert loaded_dev.api_keys[0].name == "Test Key"


@pytest.mark.asyncio
async def test_api_key_unique_hash(db_session: AsyncSession):
    """Verify unique constraint on API key hash."""
    dev = Developer(email="carol@example.com", name="Carol Dev")
    db_session.add(dev)
    await db_session.commit()
    await db_session.refresh(dev)

    key1 = APIKey(
        developer_id=dev.id,
        name="Key 1",
        key_hash="same_hash_value" + "0" * 49,
        key_prefix="nlp_live_1111",
    )
    db_session.add(key1)
    await db_session.commit()

    key2 = APIKey(
        developer_id=dev.id,
        name="Key 2",
        key_hash="same_hash_value" + "0" * 49,
        key_prefix="nlp_live_2222",
    )
    db_session.add(key2)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_api_key_revocation_and_expiration(db_session: AsyncSession):
    """Verify key revocation status and expiration timestamps."""
    dev = Developer(email="dave@example.com", name="Dave Dev")
    db_session.add(dev)
    await db_session.commit()
    await db_session.refresh(dev)

    now = datetime.now(timezone.utc)
    key = APIKey(
        developer_id=dev.id,
        name="Expiring Key",
        key_hash="b" * 64,
        key_prefix="nlp_live_exp1",
        expires_at=now + timedelta(days=30),
    )
    db_session.add(key)
    await db_session.commit()
    await db_session.refresh(key)

    assert key.is_active is True
    assert key.expires_at is not None

    # Revoke key
    key.is_active = False
    key.revoked_at = datetime.now(timezone.utc)
    await db_session.commit()
    await db_session.refresh(key)

    assert key.is_active is False
    assert key.revoked_at is not None


@pytest.mark.asyncio
async def test_create_usage_record(db_session: AsyncSession):
    """Verify UsageRecord persistence, relationships, and cost units."""
    dev = Developer(email="eve@example.com", name="Eve Dev")
    db_session.add(dev)
    await db_session.commit()
    await db_session.refresh(dev)

    key = APIKey(
        developer_id=dev.id,
        name="Eve Key",
        key_hash="c" * 64,
        key_prefix="nlp_live_eve1",
    )
    db_session.add(key)
    await db_session.commit()
    await db_session.refresh(key)

    record = UsageRecord(
        developer_id=dev.id,
        api_key_id=key.id,
        endpoint="/api/v1/sentiment",
        method="POST",
        status_code=200,
        input_character_count=120,
        latency_ms=14.5,
        cost_units=1,
    )
    db_session.add(record)
    await db_session.commit()
    await db_session.refresh(record)

    assert record.id is not None
    assert record.developer_id == dev.id
    assert record.api_key_id == key.id
    assert record.endpoint == "/api/v1/sentiment"
    assert record.cost_units == 1
    assert record.timestamp is not None


@pytest.mark.asyncio
async def test_cascade_delete_developer(db_session: AsyncSession):
    """Verify deleting a developer cascades to delete API keys and usage records."""
    dev = Developer(email="frank@example.com", name="Frank Dev")
    db_session.add(dev)
    await db_session.commit()
    await db_session.refresh(dev)

    key = APIKey(
        developer_id=dev.id,
        name="Frank Key",
        key_hash="d" * 64,
        key_prefix="nlp_live_frk1",
    )
    db_session.add(key)
    await db_session.commit()
    await db_session.refresh(key)

    record = UsageRecord(
        developer_id=dev.id,
        api_key_id=key.id,
        endpoint="/api/v1/ner",
        method="POST",
        status_code=200,
        input_character_count=50,
        latency_ms=22.1,
    )
    db_session.add(record)
    await db_session.commit()

    # Delete developer
    await db_session.delete(dev)
    await db_session.commit()

    # Verify keys and usage records are deleted via cascade
    key_res = await db_session.execute(select(APIKey).where(APIKey.developer_id == dev.id))
    assert key_res.scalars().all() == []

    rec_res = await db_session.execute(select(UsageRecord).where(UsageRecord.developer_id == dev.id))
    assert rec_res.scalars().all() == []


@pytest.mark.asyncio
async def test_delete_api_key_preserves_usage_record(db_session: AsyncSession):
    """Verify deleting an APIKey sets api_key_id=NULL on usage records to preserve audit trail."""
    dev = Developer(email="grace@example.com", name="Grace Dev")
    db_session.add(dev)
    await db_session.commit()
    await db_session.refresh(dev)

    key = APIKey(
        developer_id=dev.id,
        name="Grace Key",
        key_hash="e" * 64,
        key_prefix="nlp_live_grc1",
    )
    db_session.add(key)
    await db_session.commit()
    await db_session.refresh(key)

    record = UsageRecord(
        developer_id=dev.id,
        api_key_id=key.id,
        endpoint="/api/v1/classification",
        method="POST",
        status_code=200,
        input_character_count=85,
        latency_ms=35.0,
    )
    db_session.add(record)
    await db_session.commit()
    await db_session.refresh(record)

    # Delete API key only
    await db_session.delete(key)
    await db_session.commit()

    # Usage record still exists, with api_key_id set to NULL
    await db_session.refresh(record)
    assert record.id is not None
    assert record.api_key_id is None
    assert record.developer_id == dev.id


@pytest.mark.asyncio
async def test_database_check_constraints(db_session: AsyncSession):
    """Verify check constraints prevent invalid quotas, status codes, and latencies."""
    # 1. Negative quota on Developer
    dev_invalid = Developer(email="badquota@example.com", name="Bad Quota", monthly_request_quota=-1)
    db_session.add(dev_invalid)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()

    # Create valid developer for usage tests
    dev = Developer(email="valid@example.com", name="Valid Dev")
    db_session.add(dev)
    await db_session.commit()
    await db_session.refresh(dev)
    dev_id = dev.id

    # 2. Negative character count
    rec_bad_chars = UsageRecord(
        developer_id=dev_id,
        endpoint="/api/v1/sentiment",
        method="POST",
        status_code=200,
        input_character_count=-10,
        latency_ms=10.0,
    )
    db_session.add(rec_bad_chars)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()

    # 3. Invalid HTTP status code (< 100 or > 599)
    rec_bad_status = UsageRecord(
        developer_id=dev_id,
        endpoint="/api/v1/sentiment",
        method="POST",
        status_code=999,
        input_character_count=10,
        latency_ms=10.0,
    )
    db_session.add(rec_bad_status)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()
