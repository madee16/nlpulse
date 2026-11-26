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


@pytest.mark.asyncio
async def test_registration_rate_limit_enforced_and_no_db_leak(async_client: AsyncClient, db_session, monkeypatch):
    """Verify public developer registration enforces per-IP 429 limits without persisting rejected rows."""
    from sqlalchemy import func, select
    from src.api import deps
    from src.core.config import Settings
    from src.core.rate_limiter import SlidingWindowRateLimiter
    from src.models.api_key import APIKey
    from src.models.developer import Developer

    test_reg_limiter = SlidingWindowRateLimiter(default_max_requests=2, default_window_seconds=60)
    orig_limiter = deps.registration_rate_limiter
    deps.registration_rate_limiter = test_reg_limiter
    monkeypatch.setattr(
        deps,
        "get_settings",
        lambda: Settings(REGISTRATION_RATE_LIMIT_REQUESTS=2, REGISTRATION_RATE_LIMIT_WINDOW_SECONDS=60),
    )
    try:
        r1 = await async_client.post("/api/v1/developers", json={"email": "reg_rl_1@example.com", "name": "Reg One"})
        r2 = await async_client.post("/api/v1/developers", json={"email": "reg_rl_2@example.com", "name": "Reg Two"})
        assert r1.status_code == 201
        assert r2.status_code == 201

        # Spoofed X-Forwarded-For must NOT bypass admission control when peer is untrusted
        r3 = await async_client.post(
            "/api/v1/developers",
            headers={"X-Forwarded-For": "198.51.100.77"},
            json={"email": "reg_rl_3@example.com", "name": "Reg Three"},
        )
        assert r3.status_code == 429
        assert "retry-after" in r3.headers

        dev_count = (await db_session.execute(select(func.count(Developer.id)))).scalar_one()
        key_count = (await db_session.execute(select(func.count(APIKey.id)))).scalar_one()
        assert dev_count == 2
        assert key_count == 2
    finally:
        deps.registration_rate_limiter = orig_limiter


@pytest.mark.asyncio
async def test_registration_trusted_proxy_and_independent_client_ips():
    """Verify right-most untrusted IP extraction under configured TRUSTED_PROXY_IPS vs untrusted peers."""
    from starlette.requests import Request
    from src.api.deps import resolve_client_ip

    def _make_request(peer_host: str, xff: str | None = None) -> Request:
        headers = [(b"x-forwarded-for", xff.encode())] if xff is not None else []
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/developers",
            "headers": headers,
            "client": (peer_host, 12345),
        }
        return Request(scope)

    # Untrusted peer cannot spoof X-Forwarded-For
    untrusted_req = _make_request("203.0.113.9", "198.51.100.50")
    assert resolve_client_ip(untrusted_req, trusted_proxies=["10.0.0.1"]) == "203.0.113.9"

    # Trusted proxy uses right-most untrusted address (ignores client-prepended spoofed IP)
    trusted_req = _make_request("10.0.0.1", "198.51.100.99, 192.0.2.44")
    assert resolve_client_ip(trusted_req, trusted_proxies=["10.0.0.0/8"]) == "192.0.2.44"

    # Chained trusted proxies peel off cleanly from the right
    chained_req = _make_request("10.0.0.1", "198.51.100.1, 192.0.2.55, 10.0.0.2")
    assert resolve_client_ip(chained_req, trusted_proxies=["10.0.0.0/8"]) == "192.0.2.55"


@pytest.mark.asyncio
async def test_registration_rate_limit_isolates_independent_client_ips(async_client: AsyncClient, monkeypatch):
    """Verify rate limits apply independently per client IP when forwarded through a trusted proxy."""
    from src.api import deps
    from src.core.config import Settings
    from src.core.rate_limiter import SlidingWindowRateLimiter

    test_reg_limiter = SlidingWindowRateLimiter(default_max_requests=1, default_window_seconds=60)
    orig_limiter = deps.registration_rate_limiter
    deps.registration_rate_limiter = test_reg_limiter
    monkeypatch.setattr(
        deps,
        "get_settings",
        lambda: Settings(
            REGISTRATION_RATE_LIMIT_REQUESTS=1,
            REGISTRATION_RATE_LIMIT_WINDOW_SECONDS=60,
            TRUSTED_PROXY_IPS=["127.0.0.1"],
        ),
    )
    try:
        r_ip1_ok = await async_client.post(
            "/api/v1/developers",
            headers={"X-Forwarded-For": "198.51.100.10"},
            json={"email": "ip1_ok@example.com", "name": "IP1 First"},
        )
        assert r_ip1_ok.status_code == 201

        r_ip1_blocked = await async_client.post(
            "/api/v1/developers",
            headers={"X-Forwarded-For": "198.51.100.10"},
            json={"email": "ip1_blocked@example.com", "name": "IP1 Second"},
        )
        assert r_ip1_blocked.status_code == 429

        r_ip2_ok = await async_client.post(
            "/api/v1/developers",
            headers={"X-Forwarded-For": "198.51.100.20"},
            json={"email": "ip2_ok@example.com", "name": "IP2 First"},
        )
        assert r_ip2_ok.status_code == 201
    finally:
        deps.registration_rate_limiter = orig_limiter
