"""High-level NLP service orchestrating model inference, latency telemetry, and usage metering."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.core.exceptions import NLPulseException
from src.core.logging import get_logger
from src.core.models.nlp import NERModel, SentimentModel, ZeroShotClassificationModel
from src.services.model_registry import ModelRegistry, model_registry
from src.services.usage_service import UsageService

logger = get_logger("nlpulse.services.nlp")
settings = get_settings()


class NLPValidationError(NLPulseException):
    """Raised when NLP payload validation fails (e.g. empty text, exceeded length, invalid labels)."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, status_code=422)


class NLPService:
    """Service orchestrating sentiment analysis, NER, zero-shot classification, and billing telemetry."""

    def __init__(
        self,
        session: AsyncSession,
        registry: Optional[ModelRegistry] = None,
    ) -> None:
        self.session = session
        self.registry = registry or model_registry
        self.usage_service = UsageService(session)

    def _validate_text(self, text: str) -> str:
        """Validate raw text input length and non-empty constraints."""
        if not text or not text.strip():
            raise NLPValidationError("Input text cannot be empty or solely whitespace.")

        clean_text = text.strip()
        if len(clean_text) > settings.NLP_MAX_TEXT_LENGTH:
            raise NLPValidationError(
                f"Input text exceeds maximum allowed length of {settings.NLP_MAX_TEXT_LENGTH} characters "
                f"(received {len(clean_text)} characters)."
            )
        return clean_text

    async def analyze_sentiment(
        self,
        text: str,
        developer_id: uuid.UUID,
        api_key_id: Optional[uuid.UUID] = None,
    ) -> Dict[str, Any]:
        """Execute sentiment analysis and meter usage upon success."""
        clean_text = self._validate_text(text)
        model_id = settings.MODEL_SENTIMENT_ID

        model = await self.registry.get_model(model_id)

        start = time.perf_counter()
        inference_result = await model.predict(clean_text)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        # Record usage strictly upon successful inference completion
        await self.usage_service.record_usage(
            developer_id=developer_id,
            api_key_id=api_key_id,
            endpoint=f"{settings.API_V1_PREFIX}/nlp/sentiment",
            method="POST",
            status_code=200,
            latency_ms=latency_ms,
            input_character_count=len(clean_text),
            cost_units=1,
        )

        return {
            **inference_result,
            "model_id": model_id,
            "latency_ms": latency_ms,
        }

    async def extract_entities(
        self,
        text: str,
        developer_id: uuid.UUID,
        api_key_id: Optional[uuid.UUID] = None,
    ) -> Dict[str, Any]:
        """Execute Named Entity Recognition and meter usage upon success."""
        clean_text = self._validate_text(text)
        model_id = settings.MODEL_NER_ID

        model = await self.registry.get_model(model_id)

        start = time.perf_counter()
        inference_result = await model.predict(clean_text)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        # Record usage strictly upon successful inference completion
        await self.usage_service.record_usage(
            developer_id=developer_id,
            api_key_id=api_key_id,
            endpoint=f"{settings.API_V1_PREFIX}/nlp/ner",
            method="POST",
            status_code=200,
            latency_ms=latency_ms,
            input_character_count=len(clean_text),
            cost_units=1,
        )

        return {
            **inference_result,
            "model_id": model_id,
            "latency_ms": latency_ms,
        }

    async def classify_zero_shot(
        self,
        text: str,
        candidate_labels: List[str],
        multi_label: bool,
        developer_id: uuid.UUID,
        api_key_id: Optional[uuid.UUID] = None,
    ) -> Dict[str, Any]:
        """Execute zero-shot text classification and meter usage upon success."""
        clean_text = self._validate_text(text)

        # Validate candidate labels
        if not candidate_labels:
            raise NLPValidationError("candidate_labels must contain at least one label.")

        # Clean and deduplicate while preserving order
        seen = set()
        clean_labels = []
        for lbl in candidate_labels:
            if not isinstance(lbl, str) or not lbl.strip():
                raise NLPValidationError("Candidate labels must be non-empty strings.")
            cleaned = lbl.strip()
            if cleaned not in seen:
                seen.add(cleaned)
                clean_labels.append(cleaned)

        if not clean_labels:
            raise NLPValidationError("candidate_labels must contain at least one valid non-empty string.")

        model_id = settings.MODEL_ZERO_SHOT_ID
        model = await self.registry.get_model(model_id)

        start = time.perf_counter()
        inference_result = await model.predict({
            "text": clean_text,
            "candidate_labels": clean_labels,
            "multi_label": multi_label,
        })
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        # Record usage strictly upon successful inference completion
        await self.usage_service.record_usage(
            developer_id=developer_id,
            api_key_id=api_key_id,
            endpoint=f"{settings.API_V1_PREFIX}/nlp/classification",
            method="POST",
            status_code=200,
            latency_ms=latency_ms,
            input_character_count=len(clean_text),
            cost_units=1,
        )

        return {
            **inference_result,
            "model_id": model_id,
            "latency_ms": latency_ms,
        }
