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


@pytest.mark.asyncio
async def test_extraction_preserves_legitimate_none_unknown_na_present_in_source(mock_session, mock_ollama_client):
    """Verify legitimate 'none', 'unknown', and 'N/A' values stated in source text are preserved for string fields."""
    mock_ollama_client.generate.return_value = {
        "response": '{"auth_mode": "none", "vendor": "Unknown", "middle_initial": "N/A", "missing_note": "N/A"}',
        "prompt_tokens": 60,
        "completion_tokens": 20,
        "total_tokens": 80,
        "total_duration_ns": 100000000,
        "model": "qwen2.5:3b",
    }

    service = ExtractionService(session=mock_session, ollama_client=mock_ollama_client)
    dev_id = uuid.uuid4()

    fields = [
        ExtractionField(name="auth_mode", type="string"),
        ExtractionField(name="vendor", type="string"),
        ExtractionField(name="middle_initial", type="string"),
        ExtractionField(name="missing_note", type="string", required=False),
    ]

    result = await service.extract_structured_data(
        text="Auth mode is none and vendor is Unknown.",
        developer_id=dev_id,
        fields=fields,
    )

    data = result["extracted_data"]
    assert data["auth_mode"] == "none"
    assert data["vendor"] == "Unknown"
    # "N/A" does NOT appear in source_text so it is treated as a placeholder and converted to None
    assert data["middle_initial"] is None
    assert data["missing_note"] is None


@pytest.mark.asyncio
async def test_extraction_nested_schema_and_union_types(mock_session, mock_ollama_client):
    """Verify nested object/array schemas, union types, bool-vs-int guard, and additionalProperties=False."""
    mock_ollama_client.generate.return_value = {
        "response": json.dumps({
            "company": {
                "name": "Acme Corp",
                "employees": None,
                "founded_year": 1998.0,
                "headquarters": {"city": "Austin", "postal_code": "N/A"},
            },
            "executives": [
                {"name": "Jane Doe", "tenure_years": "7", "active": "false"},
                {"name": "John Smith", "tenure_years": "unknown", "active": "N/A"},
            ],
            "tags": ["enterprise", "N/A", "saas"],
            "nullable_count": "1,250",
            "fabricated_extra": "should_be_dropped",
        }),
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "total_duration_ns": 200000000,
        "model": "qwen2.5:3b",
    }

    service = ExtractionService(session=mock_session, ollama_client=mock_ollama_client)
    dev_id = uuid.uuid4()

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "company": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "employees": {"type": ["integer", "null"]},
                    "founded_year": {"type": "integer"},
                    "headquarters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "postal_code": {"type": ["string", "null"]},
                        },
                    },
                },
            },
            "executives": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "tenure_years": {"type": ["integer", "null"]},
                        "active": {"type": ["boolean", "null"]},
                    },
                },
            },
            "tags": {"type": "array", "items": {"type": "string"}},
            "nullable_count": {"type": ["integer", "null"]},
        },
    }

    result = await service.extract_structured_data(
        text="Acme Corp was founded in 1998 in Austin. Jane Doe served 7 years.",
        developer_id=dev_id,
        schema_definition=schema,
    )

    data = result["extracted_data"]
    assert "fabricated_extra" not in data
    assert data["company"]["name"] == "Acme Corp"
    assert data["company"]["employees"] is None
    assert data["company"]["founded_year"] == 1998
    assert isinstance(data["company"]["founded_year"], int)
    assert data["company"]["headquarters"]["city"] == "Austin"
    assert data["company"]["headquarters"]["postal_code"] is None
    assert data["executives"][0] == {"name": "Jane Doe", "tenure_years": 7, "active": False}
    assert data["executives"][1] == {"name": "John Smith", "tenure_years": None, "active": None}
    assert data["tags"] == ["enterprise", "saas"]
    assert data["nullable_count"] == 1250


@pytest.mark.asyncio
async def test_extraction_rejects_malformed_schema_definition(mock_session, mock_ollama_client):
    """Verify invalid schema_definition raises ExtractionValidationError before calling Ollama."""
    service = ExtractionService(session=mock_session, ollama_client=mock_ollama_client)
    dev_id = uuid.uuid4()

    with pytest.raises(ExtractionValidationError):
        await service.extract_structured_data(
            text="Valid text",
            developer_id=dev_id,
            schema_definition={"type": "object", "properties": ["not-a-dict"]},
        )

    with pytest.raises(ExtractionValidationError):
        await service.extract_structured_data(
            text="Valid text",
            developer_id=dev_id,
            schema_definition={"type": "string"},
        )

    with pytest.raises(ExtractionValidationError):
        await service.extract_structured_data(
            text="Valid text",
            developer_id=dev_id,
            schema_definition={"type": "object", "properties": {"age": {"type": "integer", "minimum": "invalid"}}},
        )

    mock_ollama_client.generate.assert_not_called()


@pytest.mark.asyncio
async def test_extraction_enforces_json_schema_constraints_and_redacts_sensitive_data(
    mock_session, mock_ollama_client, caplog
):
    """Verify Draft 2020-12 constraints (required, types, min/max, length, pattern, format, enum, nested) and redaction."""
    service = ExtractionService(session=mock_session, ollama_client=mock_ollama_client)
    dev_id = uuid.uuid4()

    def _mock_resp(payload: dict | str):
        return {
            "response": payload if isinstance(payload, str) else json.dumps(payload),
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70,
            "total_duration_ns": 100000000,
            "model": "qwen2.5:3b",
        }

    strict_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["company", "code", "count", "contact_email", "status"],
        "properties": {
            "company": {"type": "string", "minLength": 2, "maxLength": 40},
            "code": {"type": "string", "pattern": "^[A-Z]{3}-[0-9]{2}$"},
            "count": {"type": "integer", "minimum": 1, "maximum": 100},
            "contact_email": {"type": "string", "format": "email"},
            "event_date": {"type": ["string", "null"], "format": "date"},
            "status": {"type": "string", "enum": ["active", "paused"]},
            "nested": {
                "type": "object",
                "required": ["id"],
                "properties": {"id": {"type": "string", "format": "uuid"}},
            },
            "scores": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
        },
    }

    # 1. Valid structured extraction passes cleanly (including nullable event_date=None)
    mock_ollama_client.generate.return_value = _mock_resp({
        "company": "Acme Corp",
        "code": "ACM-42",
        "count": 10,
        "contact_email": "ops@acme.example",
        "event_date": None,
        "status": "active",
        "nested": {"id": "123e4567-e89b-12d3-a456-426614174000"},
        "scores": [0.85],
    })
    ok = await service.extract_structured_data(
        text="Acme Corp ACM-42 has 10 active units, contact ops@acme.example, id 123e4567-e89b-12d3-a456-426614174000, score 0.85.",
        developer_id=dev_id,
        schema_definition=strict_schema,
    )
    assert ok["extracted_data"]["count"] == 10
    assert ok["extracted_data"]["event_date"] is None

    # 2. Invalid payloads covering all schema rules MUST fail 422 and NEVER leak canary values
    canary = "SENSITIVE_PII_TOKEN_XYZ_987"
    invalid_cases = [
        # Missing required field
        {"company": "Acme Corp", "code": "ACM-42", "count": 10, "contact_email": "ops@acme.example"},
        # Incorrect primitive type (bool for integer must reject rather than silently turn None)
        {"company": "Acme Corp", "code": "ACM-42", "count": True, "contact_email": "ops@acme.example", "status": "active"},
        # Numeric minimum / maximum
        {"company": "Acme Corp", "code": "ACM-42", "count": 0, "contact_email": "ops@acme.example", "status": "active"},
        {"company": "Acme Corp", "code": "ACM-42", "count": 999, "contact_email": "ops@acme.example", "status": "active"},
        # String minLength / maxLength
        {"company": "A", "code": "ACM-42", "count": 5, "contact_email": "ops@acme.example", "status": "active"},
        {"company": "X" * 45 + canary, "code": "ACM-42", "count": 5, "contact_email": "ops@acme.example", "status": "active"},
        # String pattern
        {"company": "Acme Corp", "code": f"bad-{canary}", "count": 5, "contact_email": "ops@acme.example", "status": "active"},
        # Format violation (email / date / uuid)
        {"company": "Acme Corp", "code": "ACM-42", "count": 5, "contact_email": f"not-an-email-{canary}", "status": "active"},
        {"company": "Acme Corp", "code": "ACM-42", "count": 5, "contact_email": "ops@acme.example", "event_date": "2026-99-99", "status": "active"},
        # Enum violation
        {"company": "Acme Corp", "code": "ACM-42", "count": 5, "contact_email": "ops@acme.example", "status": f"bad_{canary}"},
        # Nested object & array minItems/bounds violations
        {
            "company": "Acme Corp",
            "code": "ACM-42",
            "count": 5,
            "contact_email": "ops@acme.example",
            "status": "active",
            "nested": {"id": "not-a-uuid"},
            "scores": [1.5],
        },
        # Placeholder on non-nullable required string
        {"company": "unknown", "code": "ACM-42", "count": 5, "contact_email": "ops@acme.example", "status": "active"},
    ]

    caplog.clear()
    for case in invalid_cases:
        mock_ollama_client.generate.return_value = _mock_resp(case)
        with pytest.raises(ExtractionValidationError) as exc_info:
            await service.extract_structured_data(
                text=f"Source containing {canary}",
                developer_id=dev_id,
                schema_definition=strict_schema,
            )
        assert exc_info.value.status_code == 422
        assert canary not in exc_info.value.message
        assert canary not in json.dumps(exc_info.value.details or {})

    # 3. Malformed JSON must never expose raw output in logs or error details
    mock_ollama_client.generate.return_value = _mock_resp(f"not valid json {canary}")
    with pytest.raises(ExtractionValidationError) as malformed_exc:
        await service.extract_structured_data(
            text="Valid text",
            developer_id=dev_id,
            schema_definition=strict_schema,
        )
    assert canary not in malformed_exc.value.message
    assert canary not in json.dumps(malformed_exc.value.details or {})
    assert canary not in caplog.text
