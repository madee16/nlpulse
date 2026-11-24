"""Service orchestrating generative information extraction and programmatic schema validation (M6)."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Union
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.core.exceptions import (
    ExtractionValidationError,
    NLPulseException,
)
from src.core.logging import get_logger
from src.schemas.extraction import ExtractionField
from src.services.ollama_client import OllamaClient
from src.services.usage_service import UsageService

logger = get_logger("nlpulse.services.extraction")
settings = get_settings()

_SYSTEM_GROUNDED_EXTRACTION_PROMPT = """You are an expert, precise data extraction engine.
Your mission is to extract structured data strictly from the provided source document according to the specified JSON Schema.

CRITICAL OPERATIONAL RULES:
1. Extract information ONLY if explicitly stated or directly inferable from the source text.
2. Under NO circumstance should you fabricate, assume, or extrapolate unstated facts.
3. If a requested property or field is not mentioned or cannot be found in the document, you MUST set its value to null (or [] for arrays).
4. Do NOT output placeholders like "N/A", "Unknown", "Not specified", or "None" — always use JSON null for absent fields.
5. Your response MUST be valid JSON adhering strictly to the schema properties. Do not prepend or append markdown code blocks, explanations, or commentary."""


class ExtractionService:
    """Service handling generative schema-grounded extraction, JSON validation, and token billing."""

    def __init__(
        self,
        session: AsyncSession,
        ollama_client: Optional[OllamaClient] = None,
    ) -> None:
        self.session = session
        self.ollama_client = ollama_client or OllamaClient()
        self.usage_service = UsageService(session)

    def _normalize_schema(
        self,
        schema_definition: Optional[Dict[str, Any]] = None,
        fields: Optional[List[Union[ExtractionField, Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        """Convert simplified fields or raw schema into a canonical JSON Schema dict."""
        if schema_definition:
            if not isinstance(schema_definition, dict):
                raise ExtractionValidationError("schema_definition must be a JSON object dictionary.")
            return schema_definition

        if not fields:
            raise ExtractionValidationError("Either schema_definition or fields must be provided.")

        properties: Dict[str, Any] = {}
        required: List[str] = []

        for f in fields:
            if isinstance(f, dict):
                name = str(f.get("name", "")).strip()
                field_type = str(f.get("type", "string")).lower()
                desc = f.get("description")
                is_req = bool(f.get("required", True))
            elif isinstance(f, ExtractionField):
                name = f.name.strip()
                field_type = f.type.lower()
                desc = f.description
                is_req = f.required
            else:
                continue

            if not name:
                continue

            prop_def: Dict[str, Any] = {"type": field_type}
            if desc:
                prop_def["description"] = desc
            properties[name] = prop_def
            if is_req:
                required.append(name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _clean_markdown_json(self, raw_text: str) -> str:
        """Strip markdown fences, leading/trailing whitespace, or commentary around JSON."""
        cleaned = raw_text.strip()
        # Strip ```json ... ``` or ``` ... ```
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            cleaned = cleaned.strip()

        # Find first '{' and last '}'
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx : end_idx + 1]

        return cleaned

    def _validate_and_sanitize_payload(
        self,
        raw_json_str: str,
        schema: Dict[str, Any],
        source_text: str,
    ) -> Dict[str, Any]:
        """Parse raw model output into JSON and enforce schema types, null defaults, and grounding."""
        cleaned_str = self._clean_markdown_json(raw_json_str)

        try:
            data = json.loads(cleaned_str)
        except json.JSONDecodeError as exc:
            logger.warning(f"Ollama generated malformed JSON: {cleaned_str}")
            raise ExtractionValidationError(
                f"Generative model failed to produce valid JSON: {exc.msg}",
                details={"raw_output": raw_json_str[:500]},
            ) from exc

        if not isinstance(data, dict):
            raise ExtractionValidationError(
                f"Extracted JSON must be a dictionary object, received {type(data).__name__}."
            )

        properties = schema.get("properties", {})
        required_fields = set(schema.get("required", []))

        sanitized: Dict[str, Any] = {}

        # 1. Process all expected properties
        for prop_name, prop_spec in properties.items():
            expected_type = prop_spec.get("type", "string")
            value = data.get(prop_name)

            # Normalize placeholder strings into null
            if isinstance(value, str):
                normalized_str = value.strip().lower()
                if normalized_str in {"null", "none", "n/a", "unknown", "not specified", "not mentioned", ""}:
                    value = None

            # Enforce presence
            if value is None:
                sanitized[prop_name] = None
                continue

            # Type validation / soft coercion
            if expected_type in {"number", "integer"}:
                if isinstance(value, (int, float)):
                    sanitized[prop_name] = value
                elif isinstance(value, str):
                    # Strip currency symbols and commas e.g. "$89.5 billion" -> handle or convert
                    try:
                        clean_num = re.sub(r"[^\d.-]", "", value)
                        if clean_num:
                            sanitized[prop_name] = float(clean_num) if "." in clean_num else int(clean_num)
                        else:
                            sanitized[prop_name] = None
                    except ValueError:
                        sanitized[prop_name] = None
                else:
                    sanitized[prop_name] = None

            elif expected_type == "boolean":
                if isinstance(value, bool):
                    sanitized[prop_name] = value
                elif isinstance(value, str):
                    sanitized[prop_name] = value.lower() in {"true", "yes", "1"}
                else:
                    sanitized[prop_name] = bool(value)

            elif expected_type == "array":
                if isinstance(value, list):
                    sanitized[prop_name] = value
                else:
                    sanitized[prop_name] = [value]

            else:
                # Default string / object
                sanitized[prop_name] = value

        # 2. Retain any extra fields present in data
        for k, v in data.items():
            if k not in sanitized:
                sanitized[k] = v

        return sanitized

    async def extract_structured_data(
        self,
        text: str,
        developer_id: uuid.UUID,
        schema_definition: Optional[Dict[str, Any]] = None,
        fields: Optional[List[Union[ExtractionField, Dict[str, Any]]]] = None,
        instructions: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        api_key_id: Optional[uuid.UUID] = None,
    ) -> Dict[str, Any]:
        """Perform grounded information extraction using local LLM, validate JSON, and meter tokens."""
        if not text or not text.strip():
            raise ExtractionValidationError("Input text cannot be empty or whitespace.")

        clean_text = text.strip()
        if len(clean_text) > settings.OLLAMA_MAX_INPUT_LENGTH:
            raise ExtractionValidationError(
                f"Input text exceeds maximum allowed length of {settings.OLLAMA_MAX_INPUT_LENGTH} characters "
                f"(received {len(clean_text)} characters)."
            )

        target_schema = self._normalize_schema(schema_definition=schema_definition, fields=fields)
        target_model = model or settings.OLLAMA_MODEL

        # Construct structured user prompt
        user_prompt_parts = [
            "TARGET JSON SCHEMA:",
            json.dumps(target_schema, indent=2),
            "",
        ]
        if instructions and instructions.strip():
            user_prompt_parts.extend([
                "SPECIFIC EXTRACTION INSTRUCTIONS:",
                instructions.strip(),
                "",
            ])

        user_prompt_parts.extend([
            "SOURCE DOCUMENT:",
            '"""',
            clean_text,
            '"""',
            "",
            "EXTRACTED JSON OBJECT:",
        ])
        user_prompt = "\n".join(user_prompt_parts)

        start_time = time.perf_counter()

        # Call local Ollama service
        ollama_result = await self.ollama_client.generate(
            prompt=user_prompt,
            system=_SYSTEM_GROUNDED_EXTRACTION_PROMPT,
            model=target_model,
            format="json",
            options={"temperature": max(0.0, min(temperature, 2.0))},
        )

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Validate and structure returned output
        extracted_data = self._validate_and_sanitize_payload(
            raw_json_str=ollama_result["response"],
            schema=target_schema,
            source_text=clean_text,
        )

        total_tokens = ollama_result["total_tokens"]
        # Cost units: 1 unit minimum + token rate (default: 5 units per 1,000 tokens)
        cost_units = max(1, int(round((total_tokens * settings.EXTRACTION_COST_PER_1K_TOKENS) / 1000)))

        # Meter usage atomically upon successful inference
        await self.usage_service.record_usage(
            developer_id=developer_id,
            api_key_id=api_key_id,
            endpoint=f"{settings.API_V1_PREFIX}/nlp/extraction",
            method="POST",
            status_code=200,
            latency_ms=latency_ms,
            input_character_count=len(clean_text),
            cost_units=cost_units,
        )

        return {
            "extracted_data": extracted_data,
            "model": ollama_result["model"],
            "prompt_tokens": ollama_result["prompt_tokens"],
            "completion_tokens": ollama_result["completion_tokens"],
            "total_tokens": total_tokens,
            "cost_units": cost_units,
            "latency_ms": latency_ms,
        }
