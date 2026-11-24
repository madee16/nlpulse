"""Unit tests for business service logic."""

import uuid
from datetime import datetime, timedelta, timezone
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import (
    AuthenticationError,
    QuotaExceededError,
    ResourceConflictError,
    ResourceNotFoundError,
)
from src.services.api_key_service import APIKeyService
from src.services.developer_service import DeveloperService
from src.services.usage_service import UsageService


@pytest.mark.asyncio
async def test_developer_service_crud(db_session: AsyncSession):
    """Verify DeveloperService creates, queries, and enforces email uniqueness."""
    service = DeveloperService(db_session)
    dev = await service.create_developer(email="jane@example.com", name="Jane Doe", tier="pro")
    assert dev.id is not None
    assert dev.monthly_request_quota == 10000
    assert dev.tier == "pro"

    # Query by email
    found = await service.get_by_email("jane@example.com")
    assert found is not None
    assert found.id == dev.id

    # Duplicate email error
    with pytest.raises(ResourceConflictError):
        await service.create_developer(email="JANE@example.com", name="Duplicate")


@pytest.mark.asyncio
async def test_api_key_service_lifecycle(db_session: AsyncSession):
    """Verify API key creation, authentication, tenant filtering, and revocation."""
    dev_service = DeveloperService(db_session)
    dev = await dev_service.create_developer(email="key_user@example.com", name="Key User")

    key_service = APIKeyService(db_session)
    api_key, raw_key = await key_service.create_key(
        developer_id=dev.id,
        name="Production Key",
        scopes=["sentiment:read", "ner:read"],
        expires_in_days=30,
    )
    assert raw_key.startswith("nlp_live_")
    assert api_key.name == "Production Key"
    assert api_key.developer_id == dev.id
    assert api_key.is_active is True

    # Authenticate valid key
    auth_key = await key_service.authenticate_raw_key(raw_key)
    assert auth_key.id == api_key.id
    assert auth_key.last_used_at is not None

    # List keys for tenant
    keys = await key_service.list_keys(dev.id)
    assert len(keys) == 1
    assert keys[0].id == api_key.id

    # Revoke key
    revoked = await key_service.revoke_key(dev.id, api_key.id)
    assert revoked.is_active is False
    assert revoked.revoked_at is not None

    # Authenticating revoked key must fail
    with pytest.raises(AuthenticationError) as exc_info:
        await key_service.authenticate_raw_key(raw_key)
    assert "revoked" in str(exc_info.value.message).lower()


@pytest.mark.asyncio
async def test_api_key_expired_rejection(db_session: AsyncSession):
    """Verify expired keys are rejected upon authentication."""
    dev_service = DeveloperService(db_session)
    dev = await dev_service.create_developer(email="exp@example.com", name="Exp User")

    key_service = APIKeyService(dev_service.session)
    api_key, raw_key = await key_service.create_key(developer_id=dev.id, name="Short Key")
    # Manually expire key in DB
    api_key.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
    await db_session.commit()

    with pytest.raises(AuthenticationError) as exc_info:
        await key_service.authenticate_raw_key(raw_key)
    assert "expired" in str(exc_info.value.message).lower()


@pytest.mark.asyncio
async def test_usage_service_quota_enforcement(db_session: AsyncSession):
    """Verify usage recording, monthly aggregation, and quota exhaustion."""
    dev_service = DeveloperService(db_session)
    # Create developer with small quota of 2 requests
    dev = await dev_service.create_developer(email="quota_user@example.com", name="Quota User")
    dev.monthly_request_quota = 2
    await db_session.commit()

    usage_service = UsageService(db_session)

    # 1st request
    await usage_service.record_usage(
        developer_id=dev.id,
        endpoint="/api/v1/sentiment",
        method="POST",
        status_code=200,
        latency_ms=15.0,
    )
    # Check within quota
    await usage_service.assert_within_quota(dev)

    # 2nd request
    await usage_service.record_usage(
        developer_id=dev.id,
        endpoint="/api/v1/sentiment",
        method="POST",
        status_code=200,
        latency_ms=12.0,
    )

    # Summary check
    summary = await usage_service.get_usage_summary(dev)
    assert summary.requests_used == 2
    assert summary.quota_remaining == 0
    assert summary.is_quota_exceeded is True

    # 3rd request attempt must raise QuotaExceededError
    with pytest.raises(QuotaExceededError):
        await usage_service.assert_within_quota(dev)
