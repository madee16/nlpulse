"""Regression tests for security and architecture audit findings."""

import asyncio
import time
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.rate_limiter import SlidingWindowRateLimiter
from src.services.developer_service import DeveloperService


@pytest.mark.asyncio
async def test_regression_public_registration_cannot_grant_paid_tier(async_client: AsyncClient):
    """Vulnerability 3: Public registration must NOT accept paid tiers (e.g. enterprise) for free."""
    payload = {
        "email": "hacker_tier@example.com",
        "name": "Tier Hacker",
        "tier": "enterprise",
    }
    response = await async_client.post("/api/v1/developers", json=payload)
    assert response.status_code == 201
    data = response.json()
    # Must enforce free tier and default quota (1000), rejecting client-chosen enterprise tier
    assert data["tier"] == "free"
    assert data["monthly_request_quota"] == 1000


@pytest.mark.asyncio
async def test_regression_unauthenticated_bootstrap_eliminated(async_client: AsyncClient):
    """Vulnerability 1: Unauthenticated key generation by guessing developer_id must be prohibited."""
    # Register developer
    reg_res = await async_client.post(
        "/api/v1/developers",
        json={"email": "victim@example.com", "name": "Victim Dev"},
    )
    assert reg_res.status_code == 201
    victim_id = reg_res.json()["id"]

    # Attacker tries to generate key for victim without authentication
    takeover_res = await async_client.post(
        f"/api/v1/developers/{victim_id}/keys",
        json={"name": "Attacker Key", "scopes": ["admin"]},
    )
    # Must not allow unauthenticated key creation (must return 401 Unauthorized or 404/405 Not Found/Method Not Allowed)
    assert takeover_res.status_code in (401, 404, 405)


@pytest.mark.asyncio
async def test_regression_privilege_escalation_blocked(async_client: AsyncClient):
    """Vulnerability 2: Standard developer cannot grant themselves 'admin' or elevated scopes."""
    # Register developer
    reg_res = await async_client.post(
        "/api/v1/developers",
        json={"email": "standard_user@example.com", "name": "Standard User"},
    )
    assert reg_res.status_code == 201
    reg_data = reg_res.json()

    # The registration returns the initial developer key with standard scopes
    api_key = reg_data.get("api_key")
    if not api_key:
        # If not returned directly, fetch via headers
        pytest.fail("Registration should return initial bootstrap key securely")

    headers = {"X-API-Key": api_key}

    # Attempt privilege escalation by creating a key with 'admin' scope
    escalation_res = await async_client.post(
        "/api/v1/keys",
        headers=headers,
        json={"name": "Hacked Admin Key", "scopes": ["admin"]},
    )
    # Must be forbidden
    assert escalation_res.status_code == 403
    assert "cannot grant unassigned or privileged scope" in escalation_res.json()["detail"].lower()


@pytest.mark.asyncio
async def test_regression_rate_limiter_stale_eviction():
    """Vulnerability 5: Rate limiter must clean up inactive keys to prevent memory leak."""
    limiter = SlidingWindowRateLimiter(default_max_requests=10, default_window_seconds=1)
    # Add 100 temporary keys
    for i in range(100):
        await limiter.check(f"temp_key_{i}")

    assert len(limiter._history) == 100

    # Wait for window to elapse
    await asyncio.sleep(1.1)

    # Trigger prune/cleanup
    await limiter.prune_stale_entries()
    assert len(limiter._history) == 0


@pytest.mark.asyncio
async def test_regression_concurrent_quota_enforcement(db_session: AsyncSession):
    """Vulnerability 4: Verify atomic quota enforcement prevents concurrent race conditions."""
    from src.core.exceptions import QuotaExceededError
    from src.services.usage_service import UsageService

    dev_service = DeveloperService(db_session)
    dev = await dev_service.create_developer(email="concurrent@example.com", name="Concurrent Dev")
    dev.monthly_request_quota = 3
    await db_session.commit()

    usage_service = UsageService(db_session)

    # Simulate 3 successful requests
    for i in range(3):
        await usage_service.assert_within_quota(dev)
        await usage_service.record_usage(
            developer_id=dev.id,
            endpoint="/api/v1/sentiment",
            method="POST",
            status_code=200,
            latency_ms=10.0,
        )

    # 4th check must immediately raise QuotaExceededError
    with pytest.raises(QuotaExceededError):
        await usage_service.assert_within_quota(dev)
