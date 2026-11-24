"""Unit tests for NLPulse health endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check_endpoint(async_client: AsyncClient):
    """Verify GET /api/v1/health returns 200 OK and expected payload."""
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "NLPulse"
    assert data["version"] == "0.1.0"
    assert "environment" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_root_health_endpoint(async_client: AsyncClient):
    """Verify GET /health fallback route returns 200 OK and expected payload."""
    response = await async_client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "NLPulse"
    assert data["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_root_endpoint(async_client: AsyncClient):
    """Verify GET / returns 200 OK with basic service info and docs link."""
    response = await async_client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["service"] == "NLPulse"
    assert "docs_url" in data or "message" in data
