"""Integration tests for POST /api/v1/nlp/extraction endpoint (M6)."""

from unittest.mock import AsyncMock
import uuid
import pytest
from httpx import AsyncClient
from sqlalchemy import select

from src.core.config import get_settings
from src.core.exceptions import (
    OllamaModelNotFoundError,
    OllamaUnavailableError,
)
from src.core.rate_limiter import rate_limiter
from src.main import app
from src.models.developer import Developer
from src.models.usage_record import UsageRecord
from src.services.ollama_client import OllamaClient

settings = get_settings()


@pytest.fixture(autouse=True)
async def reset_limiter_and_setup_ollama_mock():
    """Reset rate limiter and provide controlled mock OllamaClient on app state."""
    await rate_limiter.reset()

    mock_ollama = AsyncMock(spec=OllamaClient)
    mock_ollama.generate.return_value = {
        "response": '{"company": "Google LLC", "ceo": "Sundar Pichai", "valuation": "$2 Trillion"}',
        "prompt_tokens": 85,
        "completion_tokens": 25,
        "total_tokens": 110,
        "total_duration_ns": 250000000,
        "model": "qwen2.5:3b",
    }
    app.state.ollama_client = mock_ollama
    yield mock_ollama


@pytest.fixture
async def registered_dev_headers(async_client: AsyncClient):
    """Register developer and return authorization headers with full-scoped key."""
    res = await async_client.post(
        "/api/v1/developers",
        json={"email": "m6_extractor@example.com", "name": "M6 Extractor Developer"},
    )
    assert res.status_code == 201
    data = res.json()
    return {"X-API-Key": data["api_key"]}, data["id"]


@pytest.mark.asyncio
async def test_extraction_api_success_200(async_client: AsyncClient, registered_dev_headers: tuple, db_session):
    """Verify POST /api/v1/nlp/extraction successfully extracts structured JSON and meters tokens."""
    headers, dev_id = registered_dev_headers

    payload = {
        "text": "Google LLC is led by CEO Sundar Pichai with a market valuation over $2 Trillion.",
        "fields": [
            {"name": "company", "type": "string", "description": "Company title"},
            {"name": "ceo", "type": "string", "description": "Chief Executive Officer"},
            {"name": "valuation", "type": "string", "description": "Market valuation"},
            {"name": "founder", "type": "string", "description": "Founding person", "required": False},
        ],
        "temperature": 0.0,
    }

    response = await async_client.post(
        "/api/v1/nlp/extraction",
        headers=headers,
        json=payload,
    )
    assert response.status_code == 200
    data = response.json()

    assert data["model"] == "qwen2.5:3b"
    assert data["prompt_tokens"] == 85
    assert data["completion_tokens"] == 25
    assert data["total_tokens"] == 110
    assert data["cost_units"] >= 1
    assert data["latency_ms"] >= 0.0

    extracted = data["extracted_data"]
    assert extracted["company"] == "Google LLC"
    assert extracted["ceo"] == "Sundar Pichai"
    assert extracted["valuation"] == "$2 Trillion"
    assert extracted["founder"] is None

    # Verify atomic DB usage record
    res = await db_session.execute(
        select(UsageRecord).where(UsageRecord.developer_id == uuid.UUID(dev_id))
    )
    records = list(res.scalars().all())
    extraction_records = [r for r in records if r.endpoint == f"{settings.API_V1_PREFIX}/nlp/extraction"]
    assert len(extraction_records) == 1
    record = extraction_records[0]
    assert record.status_code == 200
    assert record.cost_units == data["cost_units"]


@pytest.mark.asyncio
async def test_extraction_api_unauthenticated_401(async_client: AsyncClient):
    """Verify POST /api/v1/nlp/extraction without X-API-Key returns 401."""
    response = await async_client.post(
        "/api/v1/nlp/extraction",
        json={"text": "Valid text", "fields": [{"name": "item"}]},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_extraction_api_insufficient_scope_403(async_client: AsyncClient, registered_dev_headers: tuple):
    """Verify API key lacking extraction:read / analyze:read receives 403 Forbidden."""
    headers, _ = registered_dev_headers

    # Issue sub-key restricted to ner:read only
    key_res = await async_client.post(
        "/api/v1/keys",
        headers=headers,
        json={"name": "NER Restricted Key", "scopes": ["ner:read"]},
    )
    assert key_res.status_code == 201
    limited_key = key_res.json()["api_key"]

    response = await async_client.post(
        "/api/v1/nlp/extraction",
        headers={"X-API-Key": limited_key},
        json={"text": "Valid text", "fields": [{"name": "item"}]},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_extraction_api_legacy_analyze_read_compatibility(async_client: AsyncClient, registered_dev_headers: tuple):
    """Verify backwards compatibility: existing API key holding analyze:read can access extraction."""
    headers, _ = registered_dev_headers

    key_res = await async_client.post(
        "/api/v1/keys",
        headers=headers,
        json={"name": "Legacy Key", "scopes": ["analyze:read"]},
    )
    assert key_res.status_code == 201
    legacy_key = key_res.json()["api_key"]

    response = await async_client.post(
        "/api/v1/nlp/extraction",
        headers={"X-API-Key": legacy_key},
        json={"text": "Apple is headquartered in Cupertino.", "fields": [{"name": "company"}]},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_extraction_api_ollama_unavailable_503(async_client: AsyncClient, registered_dev_headers: tuple):
    """Verify 503 Service Unavailable returned when Ollama daemon cannot be reached."""
    headers, _ = registered_dev_headers
    app.state.ollama_client.generate.side_effect = OllamaUnavailableError("http://localhost:11434")

    response = await async_client.post(
        "/api/v1/nlp/extraction",
        headers=headers,
        json={"text": "Sample text", "fields": [{"name": "item"}]},
    )
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_extraction_api_model_not_found_404(async_client: AsyncClient, registered_dev_headers: tuple):
    """Verify 404 Not Found returned when specified Ollama model is not installed."""
    headers, _ = registered_dev_headers
    app.state.ollama_client.generate.side_effect = OllamaModelNotFoundError("llama3.2:3b")

    response = await async_client.post(
        "/api/v1/nlp/extraction",
        headers=headers,
        json={"text": "Sample text", "fields": [{"name": "item"}], "model": "llama3.2:3b"},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_extraction_api_quota_exhaustion_429(async_client: AsyncClient, registered_dev_headers: tuple, db_session):
    """Verify exhausted monthly quota blocks extraction request with 429."""
    headers, dev_id = registered_dev_headers

    dev = await db_session.get(Developer, uuid.UUID(dev_id))
    assert dev is not None
    dev.monthly_request_quota = 0
    await db_session.commit()

    response = await async_client.post(
        "/api/v1/nlp/extraction",
        headers=headers,
        json={"text": "Sample text", "fields": [{"name": "item"}]},
    )
    assert response.status_code == 429
    assert "quota" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_extraction_api_missing_schema_and_fields_422(async_client: AsyncClient, registered_dev_headers: tuple):
    """Verify request with neither schema_definition nor fields returns 422 Unprocessable Entity."""
    headers, _ = registered_dev_headers

    response = await async_client.post(
        "/api/v1/nlp/extraction",
        headers=headers,
        json={"text": "Sample text"},
    )
    assert response.status_code == 422
