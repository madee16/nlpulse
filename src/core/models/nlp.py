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


def _is_hf_pipeline(pipe: Any) -> bool:
    """Detect whether a pipeline object is a full Hugging Face Pipeline supporting batched lists."""
    return hasattr(pipe, "tokenizer") and hasattr(pipe, "model")


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

    def _format_sentiment_items(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert raw pipeline label-score dicts into canonical SentimentResponse payload."""
        if not items:
            raise ValueError("Sentiment pipeline returned an empty score list.")
        scores: Dict[str, float] = {
            item["label"]: round(float(item["score"]), 4) for item in items
        }
        top_item = max(items, key=lambda x: x["score"])
        return {
            "label": top_item["label"],
            "confidence": round(float(top_item["score"]), 4),
            "scores": scores,
        }

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

        return self._format_sentiment_items(items)

    def _execute_batch_inference(self, inputs: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        """Execute genuine tensor-batched sentiment analysis for a list of strings."""
        if not isinstance(inputs, list) or not all(isinstance(t, str) for t in inputs):
            raise ValueError("Expected list of strings for batched sentiment analysis.")
        if not inputs:
            return []

        batch_size = max(1, int(kwargs.get("batch_size", settings.BATCH_INFERENCE_CHUNK_SIZE)))
        if _is_hf_pipeline(self._pipeline):
            raw_output = self._pipeline(inputs, batch_size=batch_size, truncation=True)
            if isinstance(raw_output, list) and len(raw_output) == len(inputs):
                return [
                    self._format_sentiment_items(item if isinstance(item, list) else [item])
                    for item in raw_output
                ]
            raise ValueError(f"Unexpected batched sentiment pipeline output: {raw_output}")

        # Mock / custom callable: attempt batched call if supported, else fall back per item
        try:
            raw_output = self._pipeline(inputs)
            if (
                isinstance(raw_output, list)
                and len(raw_output) == len(inputs)
                and all(isinstance(el, list) for el in raw_output)
            ):
                return [self._format_sentiment_items(el) for el in raw_output]
        except (AttributeError, TypeError, ValueError):
            pass

        return [self._execute_inference(text) for text in inputs]


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

    def _format_ner_entities(self, text: str, raw_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize raw token-classification output spans into canonical NER payload."""
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
            if not word and 0 <= start <= end <= len(text):
                word = text[start:end].strip()

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
        return self._format_ner_entities(inputs, raw_entities)

    def _execute_batch_inference(self, inputs: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        """Execute batched NER extraction across a list of texts."""
        if not isinstance(inputs, list) or not all(isinstance(t, str) for t in inputs):
            raise ValueError("Expected list of strings for batched NER.")
        if not inputs:
            return []

        batch_size = max(1, int(kwargs.get("batch_size", settings.BATCH_INFERENCE_CHUNK_SIZE)))
        if _is_hf_pipeline(self._pipeline):
            raw_output = self._pipeline(inputs, batch_size=batch_size)
            if len(inputs) == 1 and isinstance(raw_output, list) and all(isinstance(el, dict) for el in raw_output):
                return [self._format_ner_entities(inputs[0], raw_output)]
            if isinstance(raw_output, list) and len(raw_output) == len(inputs):
                return [self._format_ner_entities(txt, ents) for txt, ents in zip(inputs, raw_output)]
            raise ValueError(f"Unexpected batched NER pipeline output: {raw_output}")

        try:
            raw_output = self._pipeline(inputs)
            if (
                isinstance(raw_output, list)
                and len(raw_output) == len(inputs)
                and all(isinstance(el, list) for el in raw_output)
            ):
                return [self._format_ner_entities(txt, ents) for txt, ents in zip(inputs, raw_output)]
        except (AttributeError, TypeError, ValueError):
            pass

        return [self._execute_inference(txt) for txt in inputs]


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

    def _format_zero_shot_output(self, raw_output: Any, multi_label: bool) -> Dict[str, Any]:
        """Sort and normalize zero-shot pipeline dict output."""
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
        return self._format_zero_shot_output(raw_output, multi_label=multi_label)

    def _execute_batch_inference(self, inputs: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        """Execute batched zero-shot NLI classification for texts sharing candidate_labels."""
        if not isinstance(inputs, dict):
            raise ValueError("Batched zero-shot input must be a dictionary with 'texts' and 'candidate_labels'")

        texts = inputs.get("texts", [])
        candidate_labels = inputs.get("candidate_labels", [])
        multi_label = bool(inputs.get("multi_label", False))

        if not isinstance(texts, list) or not all(isinstance(t, str) and t for t in texts):
            raise ValueError("Batched zero-shot 'texts' must be a list of non-empty strings")
        if not candidate_labels:
            raise ValueError("Candidate labels list cannot be empty")
        if not texts:
            return []

        batch_size = max(1, int(kwargs.get("batch_size", settings.BATCH_INFERENCE_CHUNK_SIZE)))
        if _is_hf_pipeline(self._pipeline):
            raw_output = self._pipeline(
                texts,
                candidate_labels=candidate_labels,
                multi_label=multi_label,
                batch_size=batch_size,
            )
            if isinstance(raw_output, dict) and len(texts) == 1:
                return [self._format_zero_shot_output(raw_output, multi_label=multi_label)]
            if isinstance(raw_output, list) and len(raw_output) == len(texts):
                return [self._format_zero_shot_output(item, multi_label=multi_label) for item in raw_output]
            raise ValueError(f"Unexpected batched zero-shot pipeline output: {raw_output}")

        try:
            raw_output = self._pipeline(
                texts,
                candidate_labels=candidate_labels,
                multi_label=multi_label,
            )
            if (
                isinstance(raw_output, list)
                and len(raw_output) == len(texts)
                and all(isinstance(el, dict) for el in raw_output)
            ):
                return [self._format_zero_shot_output(el, multi_label=multi_label) for el in raw_output]
        except (AttributeError, TypeError, ValueError):
            pass

        return [
            self._execute_inference(
                {"text": t, "candidate_labels": candidate_labels, "multi_label": multi_label}
            )
            for t in texts
        ]


class EmbeddingModel(BaseModel):
    """Embedding model computing normalized mean-pooled dense vectors (all-MiniLM-L6-v2)."""

    def __init__(
        self,
        metadata: Optional[ModelMetadata] = None,
        pipeline_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        if metadata is None:
            metadata = ModelMetadata(
                model_id=settings.MODEL_EMBEDDING_ID,
                name="MiniLM Sentence Embeddings",
                task="embedding",
                vram_estimate_mb=90.0,
                ram_estimate_mb=120.0,
                max_concurrency=settings.MODEL_DEFAULT_CONCURRENCY,
                timeout_seconds=settings.MODEL_INFERENCE_TIMEOUT_SECONDS,
            )
        super().__init__(metadata)
        self.pipeline_factory = pipeline_factory
        self._tokenizer = None
        self._model = None
        self._device_str = "cpu"

    def _load_weights(self, device: str) -> None:
        if self.pipeline_factory:
            self._model = self.pipeline_factory()
            self._device_str = device
            return

        from transformers import AutoModel, AutoTokenizer

        self._device_str = device
        logger.info(
            f"Loading embedding model '{self.metadata.model_id}' onto {device}",
            extra={"model_id": self.metadata.model_id, "device": device},
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.metadata.model_id)
        model = AutoModel.from_pretrained(self.metadata.model_id)
        if device == "cuda":
            model = model.cuda()
        else:
            model = model.cpu()
        model.eval()
        self._model = model

    def _unload_weights(self) -> None:
        self._tokenizer = None
        self._model = None
        gc.collect()

    def _execute_inference(self, inputs: Any, **kwargs: Any) -> Any:
        """Compute L2-normalized embeddings for a list of strings (or single string).

        Args:
            inputs: String or List of strings.

        Returns:
            Numpy array of shape (N, D) containing unit-normalized dense embeddings.
        """
        import numpy as np

        if isinstance(inputs, str):
            texts = [inputs]
        elif isinstance(inputs, list):
            texts = inputs
        else:
            raise ValueError(f"Expected str or List[str] for embedding input, got {type(inputs).__name__}")

        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        # Handle mock factory
        if callable(self._model) and not hasattr(self._model, "eval"):
            return self._model(texts)

        import torch

        # Tokenize with explicit length bounding respecting Sentence Transformers all-MiniLM-L6-v2 limit
        max_seq_len = 256
        if hasattr(self._tokenizer, "model_max_length") and isinstance(self._tokenizer.model_max_length, int):
            max_seq_len = min(self._tokenizer.model_max_length, 256)

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        if self._device_str == "cuda" and torch.cuda.is_available():
            encoded = {k: v.cuda() for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self._model(**encoded)
            token_embeddings = outputs[0]
            attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return normalized.cpu().numpy()


class SummarizationModel(BaseModel):
    """Abstractive text summarization using sequence-to-sequence transformers."""

    def __init__(
        self,
        metadata: Optional[ModelMetadata] = None,
        pipeline_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        if metadata is None:
            metadata = ModelMetadata(
                model_id=settings.MODEL_SUMMARIZATION_ID,
                name="DistilBART CNN Summarization",
                task="summarization",
                vram_estimate_mb=600.0,
                ram_estimate_mb=1200.0,
                max_concurrency=settings.MODEL_DEFAULT_CONCURRENCY,
                timeout_seconds=settings.MODEL_INFERENCE_TIMEOUT_SECONDS,
            )
        super().__init__(metadata)
        self.pipeline_factory = pipeline_factory
        self._pipeline = None

    def _load_weights(self, device: str) -> None:
        if self.pipeline_factory:
            self._pipeline = self.pipeline_factory()
            return

        from transformers import pipeline

        device_arg = 0 if device == "cuda" else -1
        logger.info(
            f"Loading summarization model weights '{self.metadata.model_id}' onto {device}",
            extra={"model_id": self.metadata.model_id, "device": device},
        )
        self._pipeline = pipeline(
            "summarization",
            model=self.metadata.model_id,
            device=device_arg,
        )

    def _unload_weights(self) -> None:
        self._pipeline = None
        gc.collect()

    def _execute_inference(self, inputs: Any, **kwargs: Any) -> Dict[str, Any]:
        """Perform text summarization forward pass.

        Args:
            inputs: Dictionary with keys 'text', optional 'max_length', 'min_length', 'do_sample'.

        Returns:
            Dictionary with 'summary_text'.
        """
        if not isinstance(inputs, dict):
            raise ValueError("Summarization input must be a dictionary with 'text' and optional length constraints")

        text = inputs.get("text", "")
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty for summarization")

        max_length = int(inputs.get("max_length", 130))
        min_length = int(inputs.get("min_length", 30))
        do_sample = bool(inputs.get("do_sample", False))

        if min_length > max_length:
            raise ValueError(f"min_length ({min_length}) cannot exceed max_length ({max_length})")

        if not do_sample and "temperature" in inputs and inputs["temperature"] is not None:
            raise ValueError("temperature is only effective when do_sample=True")

        pipe_kwargs: Dict[str, Any] = {
            "max_length": max_length,
            "min_length": min_length,
            "do_sample": do_sample,
            "truncation": True,
        }
        if "temperature" in inputs and inputs["temperature"] is not None:
            pipe_kwargs["temperature"] = float(inputs["temperature"])

        raw_output = self._pipeline(text, **pipe_kwargs)

        if isinstance(raw_output, list) and len(raw_output) > 0 and isinstance(raw_output[0], dict):
            summary_text = raw_output[0].get("summary_text", "").strip()
        elif isinstance(raw_output, dict):
            summary_text = raw_output.get("summary_text", "").strip()
        elif isinstance(raw_output, str):
            summary_text = raw_output.strip()
        else:
            raise ValueError(f"Unexpected summarization pipeline return format: {type(raw_output).__name__}")

        return {"summary_text": summary_text}
