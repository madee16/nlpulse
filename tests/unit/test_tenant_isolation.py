"""Unit tests verifying strict tenant isolation between developers."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.api_key import APIKey
from src.models.developer import Developer
from src.models.usage_record import UsageRecord


@pytest.mark.asyncio
async def test_tenant_isolation_api_keys(db_session: AsyncSession):
    """Verify that queries scoped to Tenant A cannot retrieve or leak Tenant B's API keys."""
    tenant_a = Developer(email="tenant_a@example.com", name="Tenant A")
    tenant_b = Developer(email="tenant_b@example.com", name="Tenant B")
    db_session.add_all([tenant_a, tenant_b])
    await db_session.commit()
    await db_session.refresh(tenant_a)
    await db_session.refresh(tenant_b)

    key_a = APIKey(
        developer_id=tenant_a.id,
        name="Key A",
        key_hash="1" * 64,
        key_prefix="nlp_live_aaaa",
    )
    key_b = APIKey(
        developer_id=tenant_b.id,
        name="Key B",
        key_hash="2" * 64,
        key_prefix="nlp_live_bbbb",
    )
    db_session.add_all([key_a, key_b])
    await db_session.commit()

    # Query scoped to Tenant A
    stmt_a = select(APIKey).where(APIKey.developer_id == tenant_a.id)
    result_a = await db_session.execute(stmt_a)
    keys_a = result_a.scalars().all()

    assert len(keys_a) == 1
    assert keys_a[0].developer_id == tenant_a.id
    assert keys_a[0].name == "Key A"

    # Verify key_b is never returned in tenant_a queries
    assert all(k.developer_id != tenant_b.id for k in keys_a)


@pytest.mark.asyncio
async def test_tenant_isolation_usage_records(db_session: AsyncSession):
    """Verify that queries scoped to Tenant A cannot retrieve or leak Tenant B's usage records."""
    tenant_a = Developer(email="tenant_a_usage@example.com", name="Tenant A Usage")
    tenant_b = Developer(email="tenant_b_usage@example.com", name="Tenant B Usage")
    db_session.add_all([tenant_a, tenant_b])
    await db_session.commit()
    await db_session.refresh(tenant_a)
    await db_session.refresh(tenant_b)

    rec_a = UsageRecord(
        developer_id=tenant_a.id,
        endpoint="/api/v1/sentiment",
        method="POST",
        status_code=200,
        input_character_count=100,
        latency_ms=10.0,
    )
    rec_b = UsageRecord(
        developer_id=tenant_b.id,
        endpoint="/api/v1/ner",
        method="POST",
        status_code=200,
        input_character_count=200,
        latency_ms=20.0,
    )
    db_session.add_all([rec_a, rec_b])
    await db_session.commit()

    # Query usage for Tenant A only
    stmt_a = select(UsageRecord).where(UsageRecord.developer_id == tenant_a.id)
    records_a = (await db_session.execute(stmt_a)).scalars().all()

    assert len(records_a) == 1
    assert records_a[0].developer_id == tenant_a.id
    assert records_a[0].endpoint == "/api/v1/sentiment"

    # Ensure Tenant B records are absent
    assert all(r.developer_id != tenant_b.id for r in records_a)
