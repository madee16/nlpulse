"""Unit tests for ExtractionService structured extraction and schema validation."""

import json
from unittest.mock import AsyncMock, MagicMock
import uuid
import pytest

from src.core.exceptions import ExtractionValidationError
from src.schemas.extraction import ExtractionField
from src.services.extraction_service import ExtractionService
from src.services.ollama_client import OllamaClient


@pytest.fixture
def mock_session():
    """Mock async SQLAlchemy database session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    return session


@pytest.fixture
def mock_ollama_client():
    """Mock OllamaClient returning controlled extraction JSON."""
    client = AsyncMock(spec=OllamaClient)
    return client


@pytest.mark.asyncio
async def test_extraction_service_success_with_fields(mock_session, mock_ollama_client):
    """Verify structured extraction using simplified field definitions."""
    mock_ollama_client.generate.return_value = {
        "response": '{"company": "Google", "revenue": "76.7 billion", "ceo": "Sundar Pichai", "location": null}',
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
        "total_duration_ns": 300000000,
        "model": "qwen2.5:3b",
    }

    service = ExtractionService(session=mock_session, ollama_client=mock_ollama_client)
    dev_id = uuid.uuid4()
    key_id = uuid.uuid4()

    fields = [
        ExtractionField(name="company", type="string", required=True),
        ExtractionField(name="revenue", type="string", required=True),
        ExtractionField(name="ceo", type="string", required=False),
        ExtractionField(name="location", type="string", required=False),
    ]

    result = await service.extract_structured_data(
        text="Google reported quarterly revenue of 76.7 billion under CEO Sundar Pichai.",
        developer_id=dev_id,
        fields=fields,
        api_key_id=key_id,
    )

    assert result["model"] == "qwen2.5:3b"
    data = result["extracted_data"]
    assert data["company"] == "Google"
    assert data["revenue"] == "76.7 billion"
    assert data["ceo"] == "Sundar Pichai"
    assert data["location"] is None

    # Check token counts and transparent cost metering
    assert result["prompt_tokens"] == 120
    assert result["completion_tokens"] == 30
    assert result["total_tokens"] == 150
    assert result["cost_units"] >= 1

    # Verify usage was recorded
    mock_session.add.assert_called_once()
    mock_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_extraction_service_success_with_json_schema(mock_session, mock_ollama_client):
    """Verify structured extraction using full standard JSON Schema specification."""
    mock_ollama_client.generate.return_value = {
        "response": '{"event": "Keynote", "attendees": 5000, "virtual": true}',
        "prompt_tokens": 90,
        "completion_tokens": 20,
        "total_tokens": 110,
        "total_duration_ns": 200000000,
        "model": "qwen2.5:3b",
    }

    service = ExtractionService(session=mock_session, ollama_client=mock_ollama_client)
    dev_id = uuid.uuid4()

    schema = {
        "type": "object",
        "properties": {
            "event": {"type": "string"},
            "attendees": {"type": "integer"},
            "virtual": {"type": "boolean"},
        },
        "required": ["event"],
    }

    result = await service.extract_structured_data(
        text="The annual developer Keynote hosted 5000 attendees virtually.",
        developer_id=dev_id,
        schema_definition=schema,
    )

    data = result["extracted_data"]
    assert data["event"] == "Keynote"
    assert data["attendees"] == 5000
    assert data["virtual"] is True


@pytest.mark.asyncio
async def test_extraction_service_null_handling_for_missing_fields(mock_session, mock_ollama_client):
    """Verify missing or placeholder fields are safely converted to null without errors."""
    mock_ollama_client.generate.return_value = {
        "response": '{"product": "iPhone", "price": "N/A", "release_year": "unknown"}',
        "prompt_tokens": 50,
        "completion_tokens": 15,
        "total_tokens": 65,
        "total_duration_ns": 150000000,
        "model": "qwen2.5:3b",
    }

    service = ExtractionService(session=mock_session, ollama_client=mock_ollama_client)
    dev_id = uuid.uuid4()

    fields = [
        ExtractionField(name="product", type="string"),
        ExtractionField(name="price", type="number"),
        ExtractionField(name="release_year", type="integer"),
        ExtractionField(name="color", type="string"),
    ]

    result = await service.extract_structured_data(
        text="Apple announced the new iPhone.",
        developer_id=dev_id,
        fields=fields,
    )

    data = result["extracted_data"]
    assert data["product"] == "iPhone"
    assert data["price"] is None
    assert data["release_year"] is None
    assert data["color"] is None


@pytest.mark.asyncio
async def test_extraction_service_markdown_fence_stripping(mock_session, mock_ollama_client):
    """Verify markdown fences (```json ... ```) are cleanly stripped from LLM output."""
    mock_ollama_client.generate.return_value = {
        "response": '```json\n{"city": "San Francisco", "state": "California"}\n```',
        "prompt_tokens": 40,
        "completion_tokens": 15,
        "total_tokens": 55,
        "total_duration_ns": 100000000,
        "model": "qwen2.5:3b",
    }

    service = ExtractionService(session=mock_session, ollama_client=mock_ollama_client)
    dev_id = uuid.uuid4()

    result = await service.extract_structured_data(
        text="The conference was held in San Francisco, California.",
        developer_id=dev_id,
        fields=[ExtractionField(name="city"), ExtractionField(name="state")],
    )

    data = result["extracted_data"]
    assert data["city"] == "San Francisco"
    assert data["state"] == "California"


@pytest.mark.asyncio
async def test_extraction_service_malformed_json_raises_validation_error(mock_session, mock_ollama_client):
    """Verify malformed non-JSON output from LLM raises ExtractionValidationError (422)."""
    mock_ollama_client.generate.return_value = {
        "response": "I could not extract any JSON from this text because it is too ambiguous.",
        "prompt_tokens": 30,
        "completion_tokens": 15,
        "total_tokens": 45,
        "total_duration_ns": 100000000,
        "model": "qwen2.5:3b",
    }

    service = ExtractionService(session=mock_session, ollama_client=mock_ollama_client)
    dev_id = uuid.uuid4()

    with pytest.raises(ExtractionValidationError) as exc_info:
        await service.extract_structured_data(
            text="Ambiguous text.",
            developer_id=dev_id,
            fields=[ExtractionField(name="entity")],
        )

    assert exc_info.value.status_code == 422
    assert "valid json" in exc_info.value.message.lower()


@pytest.mark.asyncio
async def test_extraction_service_empty_text_raises_validation_error(mock_session, mock_ollama_client):
    """Verify empty text raises ExtractionValidationError."""
    service = ExtractionService(session=mock_session, ollama_client=mock_ollama_client)
    dev_id = uuid.uuid4()

    with pytest.raises(ExtractionValidationError):
        await service.extract_structured_data(
            text="",
            developer_id=dev_id,
            fields=[ExtractionField(name="entity")],
        )
