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
    """Verify POST /api/v1/developers registers a new tenant."""
    payload = {
        "email": "dev1@example.com",
        "name": "Developer One",
        "tier": "pro",
    }
    response = await async_client.post("/api/v1/developers", json=payload)
    assert response.status_code == 201

    data = response.json()
    assert data["email"] == "dev1@example.com"
    assert data["name"] == "Developer One"
    assert data["tier"] == "pro"
    assert data["monthly_request_quota"] == 10000
    assert "id" in data


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
    """Verify creating a developer, generating an API key, and using it for authenticated requests."""
    # 1. Register developer
    dev_res = await async_client.post(
        "/api/v1/developers",
        json={"email": "keyflow@example.com", "name": "Key Flow Dev"},
    )
    dev_id = dev_res.json()["id"]

    # 2. Issue initial bootstrap key (or key created for this developer)
    key_res = await async_client.post(
        f"/api/v1/developers/{dev_id}/keys",
        json={"name": "Initial Key", "scopes": ["analyze:read"]},
    )
    assert key_res.status_code == 201
    key_data = key_res.json()
    raw_api_key = key_data["api_key"]
    key_id = key_data["id"]
    assert raw_api_key.startswith("nlp_live_")

    headers = {"X-API-Key": raw_api_key}

    # 3. Access GET /api/v1/developers/me
    me_res = await async_client.get("/api/v1/developers/me", headers=headers)
    assert me_res.status_code == 200
    assert me_res.json()["email"] == "keyflow@example.com"

    # 4. Create another API key using authenticated endpoint POST /api/v1/keys
    key2_res = await async_client.post(
        "/api/v1/keys",
        headers=headers,
        json={"name": "Secondary Key", "scopes": ["sentiment:read"]},
    )
    assert key2_res.status_code == 201
    assert "api_key" in key2_res.json()

    # 5. List keys for developer
    list_res = await async_client.get("/api/v1/keys", headers=headers)
    assert list_res.status_code == 200
    keys = list_res.json()
    assert len(keys) == 2
    # Ensure raw secret is never returned in list
    assert all("api_key" not in k for k in keys)

    # 6. Revoke the first key
    del_res = await async_client.delete(f"/api/v1/keys/{key_id}", headers=headers)
    assert del_res.status_code == 200
    assert del_res.json()["is_active"] is False

    # 7. Attempting to use the revoked key must return 401
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
    # Register and create key with only 'sentiment:read'
    dev = (await async_client.post(
        "/api/v1/developers",
        json={"email": "scope@example.com", "name": "Scope Dev"},
    )).json()

    key_res = await async_client.post(
        f"/api/v1/developers/{dev['id']}/keys",
        json={"name": "Limited Key", "scopes": ["sentiment:read"]},
    )
    raw_key = key_res.json()["api_key"]
    headers = {"X-API-Key": raw_key}

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
        json={"email": "usage_api@example.com", "name": "Usage Dev", "tier": "free"},
    )).json()

    key_res = await async_client.post(
        f"/api/v1/developers/{dev['id']}/keys",
        json={"name": "Usage Key", "scopes": ["analyze:read"]},
    )
    headers = {"X-API-Key": key_res.json()["api_key"]}

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
    key_a = (await async_client.post(
        f"/api/v1/developers/{dev_a['id']}/keys",
        json={"name": "Key A"},
    )).json()

    # Create Tenant B
    dev_b = (await async_client.post(
        "/api/v1/developers",
        json={"email": "tenant_b_api@example.com", "name": "Tenant B"},
    )).json()
    key_b = (await async_client.post(
        f"/api/v1/developers/{dev_b['id']}/keys",
        json={"name": "Key B"},
    )).json()

    # Tenant A attempts to delete Tenant B's key
    res = await async_client.delete(
        f"/api/v1/keys/{key_b['id']}",
        headers={"X-API-Key": key_a["api_key"]},
    )
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_rate_limiting_http(async_client: AsyncClient):
    """Verify exceeding process-local rate limit returns 429 and Retry-After header."""
    dev = (await async_client.post(
        "/api/v1/developers",
        json={"email": "ratelimit_http@example.com", "name": "Rate Limit Dev"},
    )).json()

    key = (await async_client.post(
        f"/api/v1/developers/{dev['id']}/keys",
        json={"name": "Rate Key", "scopes": ["analyze:read"]},
    )).json()
    headers = {"X-API-Key": key["api_key"]}

    # Default is 60 rpm. We simulate hitting limit by calling check directly or setting temporary limit
    # Test limiter with 3 requests max
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
        json={"email": "quota_http@example.com", "name": "Quota HTTP Dev"},
    )).json()

    key = (await async_client.post(
        f"/api/v1/developers/{dev['id']}/keys",
        json={"name": "Quota Key", "scopes": ["analyze:read"]},
    )).json()
    headers = {"X-API-Key": key["api_key"]}

    # Simulate exhausting developer's quota by updating monthly_request_quota to 1
    # First request consumes 1
    res1 = await async_client.get("/api/v1/protected/sentiment", headers=headers)
    assert res1.status_code == 200

    # Simulate quota exceeded by recording requests or creating a developer with 1 quota
    # Let's create developer with 1 quota
    dev_small = (await async_client.post(
        "/api/v1/developers",
        json={"email": "quota_small@example.com", "name": "Small Quota Dev"},
    )).json()
    key_small = (await async_client.post(
        f"/api/v1/developers/{dev_small['id']}/keys",
        json={"name": "Small Key", "scopes": ["analyze:read"]},
    )).json()
    headers_small = {"X-API-Key": key_small["api_key"]}

    # Record 1000 requests for this developer to surpass quota
    from src.db.session import get_db
    from src.services.usage_service import UsageService
    from uuid import UUID

    async for db in async_client._transport.app.dependency_overrides[get_db]():
        usage_svc = UsageService(db)
        for _ in range(1001):
            await usage_svc.record_usage(
                developer_id=UUID(dev_small["id"]),
                endpoint="/api/v1/protected/sentiment",
                method="GET",
                status_code=200,
                latency_ms=1.0,
            )
        break

    res_blocked = await async_client.get("/api/v1/protected/sentiment", headers=headers_small)
    assert res_blocked.status_code == 429
    assert "quota" in res_blocked.json()["detail"].lower()
