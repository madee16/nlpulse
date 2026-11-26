"""Service orchestrating generative information extraction and programmatic schema validation (M6)."""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
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


_VALID_SCHEMA_TYPES = frozenset({"string", "number", "integer", "boolean", "array", "object", "null"})
_ALWAYS_PLACEHOLDERS = frozenset({"", "null"})
_CONTEXTUAL_PLACEHOLDERS = frozenset({"none", "n/a", "unknown", "not specified", "not mentioned"})
_NUMERIC_PATTERN = re.compile(r"^\s*[$€£¥]?\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*%?\s*$")


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

    def _resolve_schema_types(self, schema_node: Dict[str, Any], path: str = "$") -> Tuple[str, bool]:
        """Resolve JSON Schema 'type' (including nullable union lists) into primary type and nullable flag."""
        raw_type = schema_node.get("type")
        if raw_type is None:
            if "properties" in schema_node:
                return "object", False
            if "items" in schema_node:
                return "array", False
            return "string", False

        if isinstance(raw_type, str):
            norm = raw_type.strip().lower()
            if norm not in _VALID_SCHEMA_TYPES:
                raise ExtractionValidationError(f"Unsupported schema type '{raw_type}' at '{path}'.")
            return norm, norm == "null"

        if isinstance(raw_type, list) and raw_type:
            normalized_types: List[str] = []
            for entry in raw_type:
                if not isinstance(entry, str) or entry.strip().lower() not in _VALID_SCHEMA_TYPES:
                    raise ExtractionValidationError(f"Invalid type entry '{entry}' in schema at '{path}'.")
                normalized_types.append(entry.strip().lower())
            allows_null = "null" in normalized_types
            non_null = [t for t in normalized_types if t != "null"]
            primary = non_null[0] if non_null else "string"
            return primary, allows_null

        raise ExtractionValidationError(f"Invalid 'type' specification at '{path}'.")

    def _validate_schema_node(self, node: Any, path: str = "$") -> None:
        """Recursively validate a JSON Schema node before dispatching LLM generation."""
        if not isinstance(node, dict):
            raise ExtractionValidationError(f"Schema definition at '{path}' must be a JSON object.")

        primary_type, _ = self._resolve_schema_types(node, path=path)
        if path == "$" and primary_type != "object":
            raise ExtractionValidationError(
                f"Top-level schema_definition 'type' must be 'object' (received '{primary_type}')."
            )

        if "properties" in node:
            props = node["properties"]
            if not isinstance(props, dict):
                raise ExtractionValidationError(f"Schema 'properties' at '{path}' must be a dictionary.")
            for prop_name, prop_spec in props.items():
                if not isinstance(prop_name, str) or not prop_name.strip():
                    raise ExtractionValidationError(f"Property names at '{path}' must be non-empty strings.")
                self._validate_schema_node(prop_spec, path=f"{path}.{prop_name}")

        if "items" in node and node["items"] is not None:
            if not isinstance(node["items"], dict):
                raise ExtractionValidationError(f"Schema 'items' at '{path}' must be a JSON Schema object.")
            self._validate_schema_node(node["items"], path=f"{path}[]")

        additional = node.get("additionalProperties")
        if isinstance(additional, dict):
            self._validate_schema_node(additional, path=f"{path}.<additional>")
        elif additional is not None and not isinstance(additional, bool):
            raise ExtractionValidationError(
                f"Schema 'additionalProperties' at '{path}' must be a boolean or schema object."
            )

    def _normalize_schema(
        self,
        schema_definition: Optional[Dict[str, Any]] = None,
        fields: Optional[List[Union[ExtractionField, Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        """Convert simplified fields or raw schema into a validated canonical JSON Schema dict."""
        if schema_definition is not None:
            if not isinstance(schema_definition, dict) or not schema_definition:
                raise ExtractionValidationError("schema_definition must be a non-empty JSON object dictionary.")
            self._validate_schema_node(schema_definition, path="$")
            props = schema_definition.get("properties")
            if not props:
                raise ExtractionValidationError("schema_definition must define at least one property under 'properties'.")
            return schema_definition

        if not fields:
            raise ExtractionValidationError("Either schema_definition or fields must be provided.")

        properties: Dict[str, Any] = {}
        required: List[str] = []

        for f in fields:
            if isinstance(f, dict):
                name = str(f.get("name", "")).strip()
                field_type = str(f.get("type", "string")).strip().lower()
                desc = f.get("description")
                is_req = bool(f.get("required", True))
            elif isinstance(f, ExtractionField):
                name = f.name.strip()
                field_type = f.type.strip().lower()
                desc = f.description
                is_req = f.required
            else:
                continue

            if not name:
                continue

            if field_type not in _VALID_SCHEMA_TYPES - {"null"}:
                raise ExtractionValidationError(
                    f"Unsupported field type '{field_type}' for field '{name}'."
                )

            prop_def: Dict[str, Any] = {"type": field_type}
            if desc:
                prop_def["description"] = desc
            properties[name] = prop_def
            if is_req:
                required.append(name)

        if not properties:
            raise ExtractionValidationError("At least one valid non-empty field must be provided.")

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

    def _is_placeholder_string(self, value: str, expected_type: str, source_text: str) -> bool:
        """Detect LLM placeholder strings while preserving genuine occurrences from source_text."""
        normalized_str = value.strip().lower()
        if normalized_str in _ALWAYS_PLACEHOLDERS:
            return True
        if normalized_str not in _CONTEXTUAL_PLACEHOLDERS:
            return False

        # Non-string fields (numbers, booleans, arrays, objects) cannot validly hold placeholder text
        if expected_type != "string":
            return True

        # For string fields, preserve values that explicitly appear in the source document
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(normalized_str)}(?![A-Za-z0-9_])"
        return re.search(pattern, source_text, flags=re.IGNORECASE) is None

    def _coerce_number(self, value: Any, expected_type: str) -> Optional[Union[int, float]]:
        """Safely coerce numeric values while rejecting booleans and non-finite floats."""
        if isinstance(value, bool):
            return None

        parsed: Optional[Union[int, float]] = None
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                return None
            parsed = value
        elif isinstance(value, str):
            match = _NUMERIC_PATTERN.match(value)
            if not match:
                return None
            clean_num = match.group(1).replace(",", "")
            try:
                parsed = float(clean_num) if "." in clean_num else int(clean_num)
            except ValueError:
                return None

        if parsed is None:
            return None

        if expected_type == "integer":
            if isinstance(parsed, float):
                if parsed.is_integer():
                    return int(parsed)
                return None
            return int(parsed)

        return parsed

    def _sanitize_value(
        self,
        value: Any,
        prop_spec: Dict[str, Any],
        source_text: str,
        field_path: str = "$",
    ) -> Any:
        """Recursively normalize placeholders and validate/coerce a value against its schema spec."""
        expected_type, _ = self._resolve_schema_types(prop_spec, path=field_path)

        if isinstance(value, str) and self._is_placeholder_string(value, expected_type, source_text):
            value = None

        if value is None:
            return None

        result: Any
        if expected_type == "object":
            if not isinstance(value, dict):
                return None
            result = self._sanitize_object(value, prop_spec, source_text, prefix=field_path)

        elif expected_type == "array":
            raw_items = value if isinstance(value, list) else [value]
            items_spec = prop_spec.get("items")
            if isinstance(items_spec, dict):
                item_type, _ = self._resolve_schema_types(items_spec, path=f"{field_path}[]")
                sanitized_items = []
                for idx, item in enumerate(raw_items):
                    cleaned_item = self._sanitize_value(
                        item, items_spec, source_text, field_path=f"{field_path}[{idx}]"
                    )
                    if cleaned_item is None and item_type != "object":
                        continue
                    sanitized_items.append(cleaned_item)
                result = sanitized_items
            else:
                result = [
                    item for item in raw_items
                    if not (isinstance(item, str) and self._is_placeholder_string(item, "string", source_text))
                ]

        elif expected_type in {"number", "integer"}:
            result = self._coerce_number(value, expected_type)

        elif expected_type == "boolean":
            if isinstance(value, bool):
                result = value
            elif isinstance(value, str):
                norm = value.strip().lower()
                if norm in {"true", "yes", "1"}:
                    result = True
                elif norm in {"false", "no", "0"}:
                    result = False
                else:
                    result = None
            elif isinstance(value, int) and value in (0, 1):
                result = bool(value)
            else:
                result = None

        else:
            # String or untyped scalar
            if isinstance(value, str):
                result = value
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                result = str(value)
            else:
                result = None

        # Enforce enum constraint if provided
        if result is not None and isinstance(prop_spec.get("enum"), list):
            allowed = prop_spec["enum"]
            if result not in allowed:
                if isinstance(result, str):
                    ci_match = next(
                        (opt for opt in allowed if isinstance(opt, str) and opt.lower() == result.lower()),
                        None,
                    )
                    result = ci_match
                else:
                    result = None

        return result

    def _sanitize_object(
        self,
        data: Dict[str, Any],
        schema_node: Dict[str, Any],
        source_text: str,
        prefix: str = "$",
    ) -> Dict[str, Any]:
        """Sanitize an object dict recursively against a JSON Schema object node."""
        properties = schema_node.get("properties", {})
        additional = schema_node.get("additionalProperties", True)
        sanitized: Dict[str, Any] = {}

        if isinstance(properties, dict):
            for prop_name, prop_spec in properties.items():
                sanitized[prop_name] = self._sanitize_value(
                    data.get(prop_name),
                    prop_spec,
                    source_text,
                    field_path=f"{prefix}.{prop_name}",
                )

        if additional is False:
            return sanitized

        for k, v in data.items():
            if k not in sanitized:
                if isinstance(additional, dict):
                    sanitized[k] = self._sanitize_value(
                        v, additional, source_text, field_path=f"{prefix}.{k}"
                    )
                else:
                    sanitized[k] = v

        return sanitized

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

        return self._sanitize_object(data, schema, source_text, prefix="$")

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
