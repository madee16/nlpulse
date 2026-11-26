"""High-level NLP service orchestrating model inference, latency telemetry, and usage metering."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.core.exceptions import NLPulseException
from src.core.logging import get_logger
from src.core.models.nlp import (
    EmbeddingModel,
    NERModel,
    SentimentModel,
    SummarizationModel,
    ZeroShotClassificationModel,
)
from src.core.nlp_utils import chunk_text_by_sentences, extract_candidate_ngrams
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

    def _validate_text(self, text: str, max_length: Optional[int] = None) -> str:
        """Validate raw text input length and non-empty constraints."""
        if not text or not text.strip():
            raise NLPValidationError("Input text cannot be empty or solely whitespace.")

        clean_text = text.strip()
        limit = max_length or settings.NLP_MAX_TEXT_LENGTH
        if len(clean_text) > limit:
            raise NLPValidationError(
                f"Input text exceeds maximum allowed length of {limit} characters "
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

    async def extract_keywords(
        self,
        text: str,
        developer_id: uuid.UUID,
        max_keywords: int = 5,
        ngram_range: Tuple[int, int] = (1, 1),
        candidates: Optional[List[str]] = None,
        api_key_id: Optional[uuid.UUID] = None,
    ) -> Dict[str, Any]:
        """Extract and rank keywords/keyphrases using dense sentence embeddings and cosine similarity."""
        clean_text = self._validate_text(text)

        if max_keywords < 1 or max_keywords > 50:
            raise NLPValidationError(f"max_keywords must be between 1 and 50 (received {max_keywords}).")

        if len(ngram_range) != 2:
            raise NLPValidationError("ngram_range must be a tuple or list of exactly two integers [min_n, max_n].")

        min_n, max_n = ngram_range
        if min_n < 1 or max_n < min_n or max_n > 3:
            raise NLPValidationError(
                f"Invalid ngram_range ({min_n}, {max_n}). Must satisfy 1 <= min_n <= max_n <= 3."
            )

        candidate_phrases = extract_candidate_ngrams(
            text=clean_text,
            ngram_range=(min_n, max_n),
            candidate_filter=candidates,
            max_candidates=settings.KEYWORD_MAX_CANDIDATES,
        )

        model_id = settings.MODEL_EMBEDDING_ID

        # Edge case: text contains zero valid candidates (e.g. only stopwords)
        if not candidate_phrases:
            start = time.perf_counter()
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            await self.usage_service.record_usage(
                developer_id=developer_id,
                api_key_id=api_key_id,
                endpoint=f"{settings.API_V1_PREFIX}/nlp/keywords",
                method="POST",
                status_code=200,
                latency_ms=latency_ms,
                input_character_count=len(clean_text),
                cost_units=1,
            )
            return {
                "keywords": [],
                "total_candidates": 0,
                "model_id": model_id,
                "latency_ms": latency_ms,
            }

        model = await self.registry.get_model(model_id)

        start = time.perf_counter()
        # Compute embeddings for document and candidate phrases in a single batch
        all_texts = [clean_text] + candidate_phrases
        embeddings = await model.predict(all_texts)

        # Normalize and compute cosine similarities
        doc_emb = embeddings[0]
        cand_embs = embeddings[1:]

        # Compute cosine similarities
        dot_products = np.dot(cand_embs, doc_emb)
        doc_norm = np.linalg.norm(doc_emb)
        cand_norms = np.linalg.norm(cand_embs, axis=1)
        sim_scores = dot_products / (np.maximum(doc_norm * cand_norms, 1e-9))
        sim_scores = np.clip(sim_scores, -1.0, 1.0)

        ranked: List[Dict[str, Any]] = [
            {"keyword": phrase, "score": round(float(score), 4)}
            for phrase, score in zip(candidate_phrases, sim_scores)
        ]
        # Deterministic sorting: descending by score, ascending by keyword for ties
        ranked.sort(key=lambda x: (-x["score"], x["keyword"]))
        top_keywords = ranked[:max_keywords]
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        # Record usage
        await self.usage_service.record_usage(
            developer_id=developer_id,
            api_key_id=api_key_id,
            endpoint=f"{settings.API_V1_PREFIX}/nlp/keywords",
            method="POST",
            status_code=200,
            latency_ms=latency_ms,
            input_character_count=len(clean_text),
            cost_units=1,
        )

        return {
            "keywords": top_keywords,
            "total_candidates": len(candidate_phrases),
            "model_id": model_id,
            "latency_ms": latency_ms,
        }

    async def summarize_text(
        self,
        text: str,
        developer_id: uuid.UUID,
        min_length: int = 30,
        max_length: int = 130,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        api_key_id: Optional[uuid.UUID] = None,
    ) -> Dict[str, Any]:
        """Generate abstractive text summary, handling long documents via chunking and consolidation."""
        clean_text = self._validate_text(text, max_length=10000)

        if min_length < 5 or max_length < min_length or max_length > 1000:
            raise NLPValidationError(
                f"Invalid length constraints: min_length={min_length}, max_length={max_length}. "
                "Must satisfy 5 <= min_length <= max_length <= 1000."
            )

        if not do_sample and temperature is not None:
            raise NLPValidationError("temperature can only be specified when do_sample=True.")

        if do_sample and temperature is not None and (temperature <= 0.0 or temperature > 2.0):
            raise NLPValidationError("temperature must be between 0.0 (exclusive) and 2.0 when do_sample=True.")

        model_id = settings.MODEL_SUMMARIZATION_ID
        model = await self.registry.get_model(model_id)

        start = time.perf_counter()
        tokenizer = getattr(getattr(model, "_pipeline", None), "tokenizer", None)
        # Chunk text if document exceeds chunk size or token threshold
        chunks = chunk_text_by_sentences(
            clean_text,
            chunk_size=settings.SUMMARIZATION_CHUNK_SIZE,
            tokenizer=tokenizer,
        )

        if len(chunks) == 1:
            # Single-pass summarization
            res = await model.predict({
                "text": chunks[0],
                "min_length": min_length,
                "max_length": max_length,
                "do_sample": do_sample,
                "temperature": temperature,
            })
            final_summary = res["summary_text"]
        else:
            # Bounded hierarchical consolidation loop with strict shrinkage & iteration cap
            chunk_max_len = max(min_length, min(max_length, 80))
            chunk_min_len = min(min_length, 20)
            max_consolidation_rounds = 5

            current_text = clean_text
            current_chunks = chunks

            for _ in range(max_consolidation_rounds):
                if len(current_chunks) <= 1:
                    break

                round_summaries = []
                for chunk in current_chunks:
                    chunk_res = await model.predict({
                        "text": chunk,
                        "min_length": chunk_min_len,
                        "max_length": chunk_max_len,
                        "do_sample": do_sample,
                        "temperature": temperature,
                    })
                    summary_part = chunk_res["summary_text"].strip()
                    if summary_part:
                        round_summaries.append(summary_part)

                consolidated_input = " ".join(round_summaries).strip()
                if not consolidated_input:
                    break

                next_chunks = chunk_text_by_sentences(
                    consolidated_input,
                    chunk_size=settings.SUMMARIZATION_CHUNK_SIZE,
                    tokenizer=tokenizer,
                )

                # Stop once reduced to a single chunk, or if no strict progress was achieved
                no_progress = (
                    len(next_chunks) >= len(current_chunks)
                    and len(consolidated_input) >= len(current_text)
                )
                current_text = consolidated_input
                current_chunks = next_chunks
                if len(current_chunks) <= 1 or no_progress:
                    break

            # Final consolidation pass (bounded if pathological input prevented full convergence)
            final_input = current_text if len(current_chunks) <= 1 else current_chunks[0]
            final_res = await model.predict({
                "text": final_input,
                "min_length": min_length,
                "max_length": max_length,
                "do_sample": do_sample,
                "temperature": temperature,
            })
            final_summary = final_res["summary_text"]

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        # Record usage: cost units proportional to chunks processed
        await self.usage_service.record_usage(
            developer_id=developer_id,
            api_key_id=api_key_id,
            endpoint=f"{settings.API_V1_PREFIX}/nlp/summarization",
            method="POST",
            status_code=200,
            latency_ms=latency_ms,
            input_character_count=len(clean_text),
            cost_units=len(chunks),
        )

        return {
            "summary": final_summary,
            "original_character_count": len(clean_text),
            "summary_character_count": len(final_summary),
            "chunks_processed": len(chunks),
            "model_id": model_id,
            "latency_ms": latency_ms,
        }
