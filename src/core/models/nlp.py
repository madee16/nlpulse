"""Concrete NLP model wrappers adhering to the BaseModel lifecycle contract.

Implements Hugging Face Transformers pipeline integrations for:
1. Sentiment Analysis (DistilBERT SST-2)
2. Named Entity Recognition (BERT-NER with subword aggregation)
3. Zero-Shot Text Classification (DistilBART MNLI with multi-label support)
"""

from __future__ import annotations

import gc
from typing import Any, Callable, Dict, List, Optional

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.models.base import BaseModel, ModelMetadata

logger = get_logger("nlpulse.core.models.nlp")
settings = get_settings()


class SentimentModel(BaseModel):
    """Sentiment analysis model using Hugging Face sequence classification."""

    def __init__(
        self,
        metadata: Optional[ModelMetadata] = None,
        pipeline_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        if metadata is None:
            metadata = ModelMetadata(
                model_id=settings.MODEL_SENTIMENT_ID,
                name="DistilBERT Sentiment Analysis",
                task="sentiment",
                vram_estimate_mb=270.0,
                ram_estimate_mb=350.0,
                max_concurrency=settings.MODEL_DEFAULT_CONCURRENCY,
                timeout_seconds=settings.MODEL_INFERENCE_TIMEOUT_SECONDS,
            )
        super().__init__(metadata)
        self.pipeline_factory = pipeline_factory
        self._pipeline = None

    def _load_weights(self, device: str) -> None:
        """Load transformers sentiment analysis pipeline."""
        if self.pipeline_factory:
            self._pipeline = self.pipeline_factory()
            return

        from transformers import pipeline

        device_arg = 0 if device == "cuda" else -1
        logger.info(
            f"Loading sentiment model weights '{self.metadata.model_id}' onto {device}",
            extra={"model_id": self.metadata.model_id, "device": device},
        )
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=self.metadata.model_id,
            device=device_arg,
            top_k=None,
        )

    def _unload_weights(self) -> None:
        """Release pipeline references and force garbage collection."""
        self._pipeline = None
        gc.collect()

    def _execute_inference(self, inputs: Any, **kwargs: Any) -> Dict[str, Any]:
        """Perform sentiment analysis forward pass.

        Args:
            inputs: Raw text string to analyze.

        Returns:
            Dictionary with predicted label, confidence, and full score distribution.
        """
        if not isinstance(inputs, str):
            raise ValueError(f"Expected string input for sentiment analysis, got {type(inputs).__name__}")

        raw_output = self._pipeline(inputs)
        # Hugging Face top_k=None returns list of list of dicts: [[{'label': ..., 'score': ...}, ...]]
        # or flat list of dicts: [{'label': ..., 'score': ...}, ...]
        if isinstance(raw_output, list) and len(raw_output) > 0 and isinstance(raw_output[0], list):
            items = raw_output[0]
        elif isinstance(raw_output, list):
            items = raw_output
        else:
            raise ValueError(f"Unexpected pipeline return format: {raw_output}")

        scores: Dict[str, float] = {
            item["label"]: round(float(item["score"]), 4) for item in items
        }
        top_item = max(items, key=lambda x: x["score"])

        return {
            "label": top_item["label"],
            "confidence": round(float(top_item["score"]), 4),
            "scores": scores,
        }


class NERModel(BaseModel):
    """Named Entity Recognition model with subword aggregation and span offsets."""

    def __init__(
        self,
        metadata: Optional[ModelMetadata] = None,
        pipeline_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        if metadata is None:
            metadata = ModelMetadata(
                model_id=settings.MODEL_NER_ID,
                name="BERT Named Entity Recognition",
                task="ner",
                vram_estimate_mb=420.0,
                ram_estimate_mb=500.0,
                max_concurrency=settings.MODEL_DEFAULT_CONCURRENCY,
                timeout_seconds=settings.MODEL_INFERENCE_TIMEOUT_SECONDS,
            )
        super().__init__(metadata)
        self.pipeline_factory = pipeline_factory
        self._pipeline = None

    def _load_weights(self, device: str) -> None:
        """Load transformers token classification pipeline with simple aggregation."""
        if self.pipeline_factory:
            self._pipeline = self.pipeline_factory()
            return

        from transformers import pipeline

        device_arg = 0 if device == "cuda" else -1
        logger.info(
            f"Loading NER model weights '{self.metadata.model_id}' onto {device}",
            extra={"model_id": self.metadata.model_id, "device": device},
        )
        self._pipeline = pipeline(
            "ner",
            model=self.metadata.model_id,
            device=device_arg,
            aggregation_strategy="simple",
        )

    def _unload_weights(self) -> None:
        """Release pipeline references and force garbage collection."""
        self._pipeline = None
        gc.collect()

    def _execute_inference(self, inputs: Any, **kwargs: Any) -> Dict[str, Any]:
        """Perform NER extraction with subword aggregation.

        Args:
            inputs: Raw text string.

        Returns:
            Dictionary containing extracted entities, offsets, and entity count.
        """
        if not isinstance(inputs, str):
            raise ValueError(f"Expected string input for NER, got {type(inputs).__name__}")

        raw_entities = self._pipeline(inputs)
        formatted_entities: List[Dict[str, Any]] = []

        for item in raw_entities:
            entity_type = item.get("entity_group", item.get("entity", "UNKNOWN"))
            start = int(item.get("start", 0))
            end = int(item.get("end", 0))

            # Guard against inverted or negative offsets
            if start < 0:
                start = 0
            if end < start:
                start, end = end, start

            word = item.get("word", "").strip()
            if not word and 0 <= start <= end <= len(inputs):
                word = inputs[start:end].strip()

            formatted_entities.append(
                {
                    "entity": entity_type,
                    "text": word,
                    "confidence": round(float(item.get("score", 0.0)), 4),
                    "start_offset": start,
                    "end_offset": end,
                }
            )

        return {
            "entities": formatted_entities,
            "entity_count": len(formatted_entities),
        }


class ZeroShotClassificationModel(BaseModel):
    """NLI-based zero-shot text classification supporting multi-label and single-label modes."""

    def __init__(
        self,
        metadata: Optional[ModelMetadata] = None,
        pipeline_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        if metadata is None:
            metadata = ModelMetadata(
                model_id=settings.MODEL_ZERO_SHOT_ID,
                name="DistilBART Zero-Shot Classification",
                task="zero-shot",
                vram_estimate_mb=550.0,
                ram_estimate_mb=700.0,
                max_concurrency=settings.MODEL_DEFAULT_CONCURRENCY,
                timeout_seconds=settings.MODEL_INFERENCE_TIMEOUT_SECONDS,
            )
        super().__init__(metadata)
        self.pipeline_factory = pipeline_factory
        self._pipeline = None

    def _load_weights(self, device: str) -> None:
        """Load transformers zero-shot classification pipeline."""
        if self.pipeline_factory:
            self._pipeline = self.pipeline_factory()
            return

        from transformers import pipeline

        device_arg = 0 if device == "cuda" else -1
        logger.info(
            f"Loading zero-shot model weights '{self.metadata.model_id}' onto {device}",
            extra={"model_id": self.metadata.model_id, "device": device},
        )
        self._pipeline = pipeline(
            "zero-shot-classification",
            model=self.metadata.model_id,
            device=device_arg,
        )

    def _unload_weights(self) -> None:
        """Release pipeline references and force garbage collection."""
        self._pipeline = None
        gc.collect()

    def _execute_inference(self, inputs: Any, **kwargs: Any) -> Dict[str, Any]:
        """Perform zero-shot classification forward pass.

        Args:
            inputs: Dictionary with keys 'text', 'candidate_labels', and optional 'multi_label'.

        Returns:
            Dictionary with sorted candidate predictions, top prediction, and classification mode.
        """
        if not isinstance(inputs, dict):
            raise ValueError("Zero-shot input must be a dictionary with 'text' and 'candidate_labels'")

        text = inputs.get("text", "")
        candidate_labels = inputs.get("candidate_labels", [])
        multi_label = bool(inputs.get("multi_label", False))

        if not text:
            raise ValueError("Text cannot be empty for zero-shot classification")
        if not candidate_labels:
            raise ValueError("Candidate labels list cannot be empty")

        raw_output = self._pipeline(
            text,
            candidate_labels=candidate_labels,
            multi_label=multi_label,
        )

        if not isinstance(raw_output, dict):
            raise ValueError(f"Unexpected pipeline return format: {type(raw_output).__name__}")

        labels = raw_output.get("labels", [])
        scores = raw_output.get("scores", [])

        predictions: List[Dict[str, Any]] = [
            {"label": lbl, "score": round(float(sc), 4)}
            for lbl, sc in zip(labels, scores)
        ]

        # Explicitly guarantee descending order by score
        predictions.sort(key=lambda x: x["score"], reverse=True)

        top_pred = predictions[0] if predictions else None

        return {
            "top_label": top_pred["label"] if top_pred else None,
            "top_score": top_pred["score"] if top_pred else None,
            "predictions": predictions,
            "multi_label": multi_label,
        }
