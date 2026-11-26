"""Pydantic schemas for generative structured information extraction (M6)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


_ALLOWED_FIELD_TYPES = frozenset({"string", "number", "integer", "boolean", "array", "object"})


class ExtractionField(BaseModel):
    """Specification of a single field to extract from document."""

    name: str = Field(..., min_length=1, max_length=100, description="Field key name in output JSON")
    type: str = Field(
        default="string",
        description="Data type of field: string, number, integer, boolean, array, or object",
    )
    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Detailed description or extraction guidance for this field",
    )
    required: bool = Field(
        default=True,
        description="Whether this field must be present in output JSON (defaults to null if not found in text)",
    )

    @field_validator("name")
    @classmethod
    def validate_name_non_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Field name cannot be empty or whitespace.")
        return cleaned

    @field_validator("type")
    @classmethod
    def validate_field_type(cls, v: str) -> str:
        normalized = v.strip().lower()
        if normalized not in _ALLOWED_FIELD_TYPES:
            raise ValueError(
                f"Unsupported field type '{v}'. Allowed types: {sorted(_ALLOWED_FIELD_TYPES)}."
            )
        return normalized


class ExtractionRequest(BaseModel):
    """Payload for local LLM-powered structured information extraction."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=15000,
        description="Source document text from which information is extracted",
        examples=[
            "Apple Inc. announced Q4 revenue of $89.5 billion with CEO Tim Cook speaking from Cupertino, California."
        ],
    )
    schema_definition: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Standard JSON Schema specification describing the desired output structure",
    )
    fields: Optional[List[ExtractionField]] = Field(
        default=None,
        description="Simplified list of field definitions to extract",
        examples=[
            [
                {"name": "company", "type": "string", "description": "Company name", "required": True},
                {"name": "revenue", "type": "string", "description": "Reported revenue", "required": True},
                {"name": "ceo", "type": "string", "description": "Chief Executive Officer", "required": False},
                {"name": "location", "type": "string", "description": "Headquarters or event location", "required": False},
            ]
        ],
    )
    instructions: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Optional additional instructions or domain-specific extraction rules",
        examples=["Extract monetary amounts including their respective currency symbols."],
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional local generative model override (e.g. 'qwen2.5:3b', 'llama3.2:3b')",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature; 0.0 guarantees deterministic, factually grounded extraction",
    )

    @model_validator(mode="after")
    def validate_schema_or_fields(self) -> "ExtractionRequest":
        if not self.schema_definition and not self.fields:
            raise ValueError(
                "Either 'schema_definition' (JSON Schema dict) or 'fields' (list of ExtractionField) must be provided."
            )
        return self


class ExtractionResponse(BaseModel):
    """Structured response containing validated extraction output and metering telemetry."""

    extracted_data: Dict[str, Any] = Field(
        ...,
        description="Validated structured data extracted from source text adhering strictly to schema",
    )
    model: str = Field(..., description="Local LLM model identifier utilized for extraction")
    prompt_tokens: int = Field(..., description="Count of prompt/context tokens evaluated")
    completion_tokens: int = Field(..., description="Count of completion tokens generated")
    total_tokens: int = Field(..., description="Total token volume processed")
    cost_units: int = Field(..., description="Transparent cost units metered against developer monthly quota")
    latency_ms: float = Field(..., description="Total execution latency in milliseconds")
