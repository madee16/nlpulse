"""Integration tests for Developer registration, API key issuance, and authentication."""

import pytest
from httpx import AsyncClient

from src.core.rate_limiter import rate_limiter


@pytest.fixture(autouse=True)
async def reset_limiter():
    """Reset rate limiter before each test."""
    await rate_limiter.reset()


@pytest.mark.asyncio
async def test_developer_registration(async_client: AsyncClient):
    """Verify POST /api/v1/developers registers a new tenant and issues bootstrap key."""
    payload = {
        "email": "dev1@example.com",
        "name": "Developer One",
    }
    response = await async_client.post("/api/v1/developers", json=payload)
    assert response.status_code == 201

    data = response.json()
    assert data["email"] == "dev1@example.com"
    assert data["name"] == "Developer One"
    assert data["tier"] == "free"
    assert data["monthly_request_quota"] == 1000
    assert "id" in data
    # Verified: plaintext key issued securely once on registration
    assert "api_key" in data
    assert data["api_key"].startswith("nlp_live_")


@pytest.mark.asyncio
async def test_developer_registration_duplicate_email(async_client: AsyncClient):
    """Verify POST /api/v1/developers rejects duplicate email with 409 Conflict."""
    payload = {"email": "dup@example.com", "name": "First"}
    res1 = await async_client.post("/api/v1/developers", json=payload)
    assert res1.status_code == 201

    res2 = await async_client.post("/api/v1/developers", json=payload)
    assert res2.status_code == 409
    assert "already registered" in res2.json()["detail"].lower()


@pytest.mark.asyncio
async def test_api_key_issuance_and_auth_flow(async_client: AsyncClient):
    """Verify registration, authenticated key creation, listing, and revocation."""
    # 1. Register developer and receive bootstrap key
    dev_res = await async_client.post(
        "/api/v1/developers",
        json={"email": "keyflow@example.com", "name": "Key Flow Dev"},
    )
    dev_data = dev_res.json()
    initial_api_key = dev_data["api_key"]
    initial_key_id = dev_data["key_id"]
    headers = {"X-API-Key": initial_api_key}

    # 2. Access GET /api/v1/developers/me
    me_res = await async_client.get("/api/v1/developers/me", headers=headers)
    assert me_res.status_code == 200
    assert me_res.json()["email"] == "keyflow@example.com"

    # 3. Create another API key using authenticated endpoint POST /api/v1/keys
    key2_res = await async_client.post(
        "/api/v1/keys",
        headers=headers,
        json={"name": "Secondary Key", "scopes": ["sentiment:read"]},
    )
    assert key2_res.status_code == 201
    assert "api_key" in key2_res.json()

    # 4. List keys for developer
    list_res = await async_client.get("/api/v1/keys", headers=headers)
    assert list_res.status_code == 200
    keys = list_res.json()
    assert len(keys) == 2
    # Ensure raw secret is never returned in list
    assert all("api_key" not in k for k in keys)

    # 5. Revoke the first key
    del_res = await async_client.delete(f"/api/v1/keys/{initial_key_id}", headers=headers)
    assert del_res.status_code == 200
    assert del_res.json()["is_active"] is False

    # 6. Attempting to use the revoked key must return 401
    failed_res = await async_client.get("/api/v1/developers/me", headers=headers)
    assert failed_res.status_code == 401
    assert "revoked" in failed_res.json()["detail"].lower()


@pytest.mark.asyncio
async def test_authentication_failures(async_client: AsyncClient):
    """Verify unauthorized requests return 401."""
    # Missing header
    res1 = await async_client.get("/api/v1/developers/me")
    assert res1.status_code == 401

    # Invalid key
    res2 = await async_client.get(
        "/api/v1/developers/me",
        headers={"X-API-Key": "nlp_live_invalidkey12345678901234567890"},
    )
    assert res2.status_code == 401


@pytest.mark.asyncio
async def test_scope_enforcement(async_client: AsyncClient):
    """Verify endpoints enforce required scopes with 403 Forbidden."""
    dev = (await async_client.post(
        "/api/v1/developers",
        json={"email": "scope@example.com", "name": "Scope Dev"},
    )).json()

    # Key issued during registration has standard scopes including sentiment:read
    headers = {"X-API-Key": dev["api_key"]}

    # Endpoint requiring 'sentiment:read' succeeds
    ok_res = await async_client.get("/api/v1/protected/sentiment", headers=headers)
    assert ok_res.status_code == 200

    # Endpoint requiring 'admin' fails with 403
    forbidden_res = await async_client.get("/api/v1/protected/admin-only", headers=headers)
    assert forbidden_res.status_code == 403
    assert "lacks required scope" in forbidden_res.json()["detail"].lower()


@pytest.mark.asyncio
async def test_usage_summary_endpoint(async_client: AsyncClient):
    """Verify GET /api/v1/usage/summary returns quota and consumption statistics."""
    dev = (await async_client.post(
        "/api/v1/developers",
        json={"email": "usage_api@example.com", "name": "Usage Dev"},
    )).json()

    headers = {"X-API-Key": dev["api_key"]}
    res = await async_client.get("/api/v1/usage/summary", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["monthly_quota"] == 1000
    assert data["requests_used"] == 0
    assert data["quota_remaining"] == 1000
    assert data["is_quota_exceeded"] is False


@pytest.mark.asyncio
async def test_tenant_cannot_revoke_other_tenant_key(async_client: AsyncClient):
    """Verify Tenant A cannot view or delete Tenant B's API key (returns 404)."""
    # Create Tenant A
    dev_a = (await async_client.post(
        "/api/v1/developers",
        json={"email": "tenant_a_api@example.com", "name": "Tenant A"},
    )).json()

    # Create Tenant B
    dev_b = (await async_client.post(
        "/api/v1/developers",
        json={"email": "tenant_b_api@example.com", "name": "Tenant B"},
    )).json()

    # Tenant A attempts to delete Tenant B's key
    res = await async_client.delete(
        f"/api/v1/keys/{dev_b['key_id']}",
        headers={"X-API-Key": dev_a["api_key"]},
    )
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_rate_limiting_http(async_client: AsyncClient):
    """Verify exceeding process-local rate limit returns 429 and Retry-After header."""
    dev = (await async_client.post(
        "/api/v1/developers",
        json={"email": "ratelimit_http@example.com", "name": "Rate Limit Dev"},
    )).json()
    headers = {"X-API-Key": dev["api_key"]}

    from src.core.rate_limiter import SlidingWindowRateLimiter
    from src.api import deps

    test_limiter = SlidingWindowRateLimiter(default_max_requests=2, default_window_seconds=10)
    orig_limiter = deps.rate_limiter
    deps.rate_limiter = test_limiter

    try:
        res1 = await async_client.get("/api/v1/protected/sentiment", headers=headers)
        assert res1.status_code == 200
        res2 = await async_client.get("/api/v1/protected/sentiment", headers=headers)
        assert res2.status_code == 200

        # 3rd request should exceed limit
        res3 = await async_client.get("/api/v1/protected/sentiment", headers=headers)
        assert res3.status_code == 429
        assert "rate limit exceeded" in res3.json()["detail"].lower()
        assert "retry-after" in res3.headers
    finally:
        deps.rate_limiter = orig_limiter


@pytest.mark.asyncio
async def test_quota_exhaustion_http(async_client: AsyncClient):
    """Verify exhausted monthly quota returns 429 Quota Exceeded."""
    dev = (await async_client.post(
        "/api/v1/developers",
        json={"email": "quota_small@example.com", "name": "Small Quota Dev"},
    )).json()
    headers = {"X-API-Key": dev["api_key"]}

    from src.db.session import get_db
    from src.services.usage_service import UsageService
    from uuid import UUID

    # Exceed the 1000 quota by recording 1001 usage records
    async for db in async_client._transport.app.dependency_overrides[get_db]():
        usage_svc = UsageService(db)
        for _ in range(1001):
            await usage_svc.record_usage(
                developer_id=UUID(dev["id"]),
                endpoint="/api/v1/protected/sentiment",
                method="GET",
                status_code=200,
                latency_ms=1.0,
            )
        break

    res_blocked = await async_client.get("/api/v1/protected/sentiment", headers=headers)
    assert res_blocked.status_code == 429
    assert "quota" in res_blocked.json()["detail"].lower()
