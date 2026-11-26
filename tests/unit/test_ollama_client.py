"""Unit tests for OllamaClient communicating with local Ollama service."""

from unittest.mock import AsyncMock, MagicMock
import httpx
import pytest

from src.core.exceptions import (
    OllamaInferenceError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from src.services.ollama_client import OllamaClient


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx.AsyncClient."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


@pytest.mark.asyncio
async def test_ollama_health_check_success(mock_httpx_client):
    """Verify health check returns True when Ollama /api/tags returns 200."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_httpx_client.get = AsyncMock(return_value=mock_resp)

    client = OllamaClient(http_client=mock_httpx_client)
    assert await client.check_health() is True
    mock_httpx_client.get.assert_awaited_once()


@pytest.mark.asyncio
async def test_ollama_health_check_unreachable(mock_httpx_client):
    """Verify health check returns False when connection to Ollama fails."""
    mock_httpx_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

    client = OllamaClient(http_client=mock_httpx_client)
    assert await client.check_health() is False


@pytest.mark.asyncio
async def test_ollama_generate_success(mock_httpx_client):
    """Verify successful Ollama generation returns parsed tokens and response string."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "qwen2.5:3b",
        "response": '{"company": "Apple Inc.", "revenue": "$89.5B"}',
        "prompt_eval_count": 45,
        "eval_count": 22,
        "total_duration": 450000000,
        "done": True,
    }
    mock_httpx_client.post = AsyncMock(return_value=mock_resp)

    client = OllamaClient(http_client=mock_httpx_client)
    result = await client.generate(
        prompt="Extract details",
        model="qwen2.5:3b",
        system="System prompt",
    )

    assert result["model"] == "qwen2.5:3b"
    assert result["prompt_tokens"] == 45
    assert result["completion_tokens"] == 22
    assert result["total_tokens"] == 67
    assert 'Apple Inc.' in result["response"]


@pytest.mark.asyncio
async def test_ollama_service_unavailable(mock_httpx_client):
    """Verify OllamaUnavailableError (503) raised when Ollama daemon is offline."""
    mock_httpx_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

    client = OllamaClient(http_client=mock_httpx_client)
    with pytest.raises(OllamaUnavailableError) as exc_info:
        await client.generate(prompt="Test prompt")

    assert exc_info.value.status_code == 503
    assert "unavailable" in exc_info.value.message.lower()


@pytest.mark.asyncio
async def test_ollama_model_not_found(mock_httpx_client):
    """Verify OllamaModelNotFoundError (404) raised when model is not downloaded."""
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.text = "model 'nonexistent-model' not found, try pulling it first"
    mock_httpx_client.post = AsyncMock(return_value=mock_resp)

    client = OllamaClient(http_client=mock_httpx_client)
    with pytest.raises(OllamaModelNotFoundError) as exc_info:
        await client.generate(prompt="Test prompt", model="nonexistent-model")

    assert exc_info.value.status_code == 404
    assert "nonexistent-model" in exc_info.value.message


@pytest.mark.asyncio
async def test_ollama_timeout(mock_httpx_client):
    """Verify OllamaTimeoutError (504) raised when inference exceeds timeout."""
    mock_httpx_client.post = AsyncMock(side_effect=httpx.TimeoutException("Read timed out"))

    client = OllamaClient(http_client=mock_httpx_client)
    with pytest.raises(OllamaTimeoutError) as exc_info:
        await client.generate(prompt="Test prompt")

    assert isinstance(exc_info.value, OllamaInferenceError)
    assert exc_info.value.status_code == 504
    assert "timed out" in exc_info.value.message.lower()


@pytest.mark.asyncio
async def test_ollama_connect_timeout_raises_unavailable_503(mock_httpx_client):
    """Verify ConnectTimeout maps to OllamaUnavailableError (503) rather than 504."""
    mock_httpx_client.post = AsyncMock(side_effect=httpx.ConnectTimeout("Connect timed out"))

    client = OllamaClient(http_client=mock_httpx_client)
    with pytest.raises(OllamaUnavailableError) as exc_info:
        await client.generate(prompt="Test prompt")

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_ollama_list_models(mock_httpx_client):
    """Verify list_models queries /api/tags and returns model list."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [{"name": "qwen2.5:3b", "size": 1900000000}]
    }
    mock_httpx_client.get = AsyncMock(return_value=mock_resp)

    client = OllamaClient(http_client=mock_httpx_client)
    models = await client.list_models()
    assert len(models) == 1
    assert models[0]["name"] == "qwen2.5:3b"


@pytest.mark.asyncio
async def test_ollama_non_json_payload_redacts_raw_response(mock_httpx_client):
    """Verify non-JSON HTTP 200 payload from Ollama raises 502 without echoing raw content."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "RAW_SENSITIVE_MODEL_OUTPUT_LEAK_777"
    mock_resp.json.side_effect = ValueError("Expecting value")
    mock_httpx_client.post = AsyncMock(return_value=mock_resp)

    client = OllamaClient(http_client=mock_httpx_client)
    with pytest.raises(OllamaInferenceError) as exc_info:
        await client.generate(prompt="Test prompt")

    assert exc_info.value.status_code == 502
    assert "RAW_SENSITIVE_MODEL_OUTPUT_LEAK_777" not in exc_info.value.message
