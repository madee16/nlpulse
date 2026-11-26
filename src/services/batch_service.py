"""Batch NLP orchestration service coordinating vectorized inference, bounded workers, and atomic metering."""

from __future__ import annotations

import asyncio
from collections import defaultdict
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, TypeVar
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.core.exceptions import (
    ExtractionValidationError,
    ModelError,
    ModelInferenceError,
    OllamaInferenceError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from src.core.logging import get_logger
from src.core.nlp_utils import extract_candidate_ngrams
from src.models.developer import Developer
from src.models.usage_record import UsageRecord
from src.schemas.batch import (
    BatchExtractionRequest,
    BatchItemError,
    BatchItemResult,
    BatchKeywordRequest,
    BatchNERRequest,
    BatchResponse,
    BatchSentimentRequest,
    BatchSummarizationRequest,
    BatchZeroShotRequest,
)
from src.schemas.extraction import ExtractionResponse
from src.schemas.nlp import (
    KeywordResponse,
    NERResponse,
    SentimentResponse,
    SummarizationResponse,
    ZeroShotResponse,
)
from src.services.extraction_service import ExtractionService
from src.services.model_registry import ModelRegistry, model_registry
from src.services.nlp_service import NLPService, NLPValidationError
from src.services.ollama_client import OllamaClient
from src.services.usage_service import UsageService

logger = get_logger("nlpulse.services.batch")
settings = get_settings()

WorkItemT = TypeVar("WorkItemT")
ResultT = TypeVar("ResultT")


def _map_exception_to_item_error(exc: Exception) -> BatchItemError:
    """Translate domain and model exceptions into safe, structured item errors."""
    if isinstance(exc, (NLPValidationError, ExtractionValidationError, ValueError)):
        msg = exc.message if hasattr(exc, "message") else str(exc)
        return BatchItemError(code="validation_error", message=msg, status_code=422)
    if isinstance(exc, OllamaUnavailableError):
        return BatchItemError(code="ollama_unavailable", message=exc.message, status_code=503)
    if isinstance(exc, OllamaModelNotFoundError):
        return BatchItemError(code="ollama_model_not_found", message=exc.message, status_code=404)
    if isinstance(exc, OllamaTimeoutError):
        return BatchItemError(code="ollama_timeout", message=exc.message, status_code=504)
    if isinstance(exc, OllamaInferenceError):
        return BatchItemError(code="ollama_inference_error", message=exc.message, status_code=502)
    if isinstance(exc, ModelInferenceError):
        return BatchItemError(code="model_inference_error", message=exc.message, status_code=500)
    if isinstance(exc, ModelError):
        return BatchItemError(code="model_error", message=exc.message, status_code=exc.status_code)

    logger.warning(f"Unexpected item processing failure in batch: {type(exc).__name__}: {exc}")
    return BatchItemError(
        code="internal_item_error",
        message="Item processing failed due to an unexpected internal error.",
        status_code=500,
    )


async def _run_bounded_workers(
    work_items: List[WorkItemT],
    concurrency: int,
    worker_fn: Callable[[WorkItemT], Awaitable[None]],
) -> None:
    """Execute work items using a fixed-size worker pool to bound both tasks and concurrency."""
    if not work_items:
        return

    queue: asyncio.Queue[WorkItemT] = asyncio.Queue()
    for item in work_items:
        queue.put_nowait(item)

    num_workers = min(len(work_items), max(1, concurrency))

    async def _worker() -> None:
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await worker_fn(item)
            finally:
                queue.task_done()

    workers = [asyncio.create_task(_worker()) for _ in range(num_workers)]
    try:
        await asyncio.gather(*workers)
    except asyncio.CancelledError:
        for w in workers:
            if not w.done():
                w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        raise


class BatchService:
    """Service orchestrating multi-item NLP inference with partial-failure isolation and quota safety."""

    def __init__(
        self,
        session: AsyncSession,
        registry: Optional[ModelRegistry] = None,
        ollama_client: Optional[OllamaClient] = None,
    ) -> None:
        self.session = session
        self.registry = registry or model_registry
        self.ollama_client = ollama_client or OllamaClient()
        self.nlp_service = NLPService(session=session, registry=self.registry)
        self.extraction_service = ExtractionService(session=session, ollama_client=self.ollama_client)
        self.usage_service = UsageService(session)

    def _build_usage_records(
        self,
        developer_id: uuid.UUID,
        api_key_id: Optional[uuid.UUID],
        endpoint: str,
        results: List[Optional[BatchItemResult[Any]]],
        clean_texts: Dict[int, str],
    ) -> List[UsageRecord]:
        """Construct UsageRecord entries strictly for succeeded items."""
        records: List[UsageRecord] = []
        for item_res in results:
            if item_res is not None and item_res.status == "success":
                records.append(
                    UsageRecord(
                        developer_id=developer_id,
                        api_key_id=api_key_id,
                        endpoint=endpoint,
                        method="POST",
                        status_code=200,
                        input_character_count=len(clean_texts.get(item_res.index, "")),
                        latency_ms=item_res.latency_ms,
                        cost_units=item_res.cost_units,
                    )
                )
        return records

    def _finalize_response(
        self,
        capability: str,
        execution_mode: str,
        model_id: Optional[str],
        results: List[Optional[BatchItemResult[ResultT]]],
        batch_start: float,
    ) -> BatchResponse[ResultT]:
        """Assemble ordered BatchResponse with aggregate success, failure, and cost metrics."""
        ordered_results: List[BatchItemResult[ResultT]] = [r for r in results if r is not None]
        succeeded_count = sum(1 for r in ordered_results if r.status == "success")
        failed_count = len(ordered_results) - succeeded_count
        total_cost_units = sum(r.cost_units for r in ordered_results if r.status == "success")
        total_latency_ms = round((time.perf_counter() - batch_start) * 1000, 2)

        return BatchResponse[ResultT](
            capability=capability,
            execution_mode=execution_mode,
            model_id=model_id,
            total_items=len(ordered_results),
            succeeded_count=succeeded_count,
            failed_count=failed_count,
            total_cost_units=total_cost_units,
            latency_ms=total_latency_ms,
            results=ordered_results,
        )

    async def _run_with_quota_guard(
        self,
        developer: Developer,
        api_key_id: Optional[uuid.UUID],
        capability: str,
        valid_count: int,
        clean_texts: Dict[int, str],
        results: List[Optional[BatchItemResult[Any]]],
        execute_fn: Callable[[], Awaitable[None]],
    ) -> None:
        """Reserve quota for valid items, execute batch work, and settle/release atomically."""
        if valid_count <= 0:
            await self.usage_service.assert_within_quota(developer)
            return

        await self.usage_service.reserve_quota(developer, units=valid_count)
        settled = False
        endpoint = f"{settings.API_V1_PREFIX}/nlp/batch/{capability}"
        try:
            await execute_fn()
            records = self._build_usage_records(
                developer_id=developer.id,
                api_key_id=api_key_id,
                endpoint=endpoint,
                results=results,
                clean_texts=clean_texts,
            )
            await self.usage_service.settle_batch_usage(
                developer_id=developer.id,
                reserved_units=valid_count,
                usage_records=records,
            )
            settled = True
        finally:
            if not settled:
                await asyncio.shield(
                    self.usage_service.release_reservation(developer.id, units=valid_count)
                )

    # -------------------------------------------------------------------------
    # Shared Micro-Batch Helper for Sequence / Token Pipelines
    # -------------------------------------------------------------------------

    async def _run_text_micro_batch(
        self,
        items: List[Any],
        developer: Developer,
        api_key_id: Optional[uuid.UUID],
        capability: str,
        model_id: str,
        response_cls: type[ResultT],
    ) -> BatchResponse[ResultT]:
        """Shared micro-batched execution path for text-only transformer models (Sentiment, NER)."""
        batch_start = time.perf_counter()
        results: List[Optional[BatchItemResult[ResultT]]] = [None] * len(items)
        valid_entries: List[Tuple[int, Optional[str], str]] = []
        clean_texts: Dict[int, str] = {}

        for idx, item in enumerate(items):
            try:
                clean_text = self.nlp_service._validate_text(item.text)
                clean_texts[idx] = clean_text
                valid_entries.append((idx, item.id, clean_text))
            except Exception as exc:
                results[idx] = BatchItemResult[response_cls](  # type: ignore[valid-type]
                    index=idx,
                    id=item.id,
                    status="error",
                    error=_map_exception_to_item_error(exc),
                )

        async def _execute() -> None:
            model = await self.registry.get_model(model_id)
            chunk_size = max(1, settings.BATCH_INFERENCE_CHUNK_SIZE)
            micro_batches = [
                valid_entries[i : i + chunk_size]
                for i in range(0, len(valid_entries), chunk_size)
            ]

            async def _process_chunk(chunk: List[Tuple[int, Optional[str], str]]) -> None:
                texts = [entry[2] for entry in chunk]
                t0 = time.perf_counter()
                try:
                    outputs = await model.predict_batch(texts, batch_size=len(texts))
                    batch_ms = (time.perf_counter() - t0) * 1000
                    per_item_ms = round(batch_ms / len(chunk), 2)
                    for (idx, item_id, _), out in zip(chunk, outputs):
                        results[idx] = BatchItemResult[response_cls](  # type: ignore[valid-type]
                            index=idx,
                            id=item_id,
                            status="success",
                            cost_units=1,
                            latency_ms=per_item_ms,
                            result=response_cls(**out, model_id=model_id, latency_ms=per_item_ms),
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if len(chunk) == 1:
                        idx, item_id, _ = chunk[0]
                        results[idx] = BatchItemResult[response_cls](  # type: ignore[valid-type]
                            index=idx,
                            id=item_id,
                            status="error",
                            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                            error=_map_exception_to_item_error(exc),
                        )
                        return
                    for idx, item_id, text in chunk:
                        item_t0 = time.perf_counter()
                        try:
                            out = await model.predict(text)
                            item_ms = round((time.perf_counter() - item_t0) * 1000, 2)
                            results[idx] = BatchItemResult[response_cls](  # type: ignore[valid-type]
                                index=idx,
                                id=item_id,
                                status="success",
                                cost_units=1,
                                latency_ms=item_ms,
                                result=response_cls(**out, model_id=model_id, latency_ms=item_ms),
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as item_exc:
                            results[idx] = BatchItemResult[response_cls](  # type: ignore[valid-type]
                                index=idx,
                                id=item_id,
                                status="error",
                                latency_ms=round((time.perf_counter() - item_t0) * 1000, 2),
                                error=_map_exception_to_item_error(item_exc),
                            )

            await _run_bounded_workers(micro_batches, settings.BATCH_MAX_CONCURRENCY, _process_chunk)

        await self._run_with_quota_guard(
            developer=developer,
            api_key_id=api_key_id,
            capability=capability,
            valid_count=len(valid_entries),
            clean_texts=clean_texts,
            results=results,
            execute_fn=_execute,
        )
        return self._finalize_response(capability, "batched_inference", model_id, results, batch_start)

    # -------------------------------------------------------------------------
    # 1. Batch Sentiment Analysis (Genuine Micro-Batched Inference)
    # -------------------------------------------------------------------------

    async def run_sentiment_batch(
        self,
        payload: BatchSentimentRequest,
        developer: Developer,
        api_key_id: Optional[uuid.UUID] = None,
    ) -> BatchResponse[SentimentResponse]:
        """Run batched sentiment analysis using micro-batched tensor inference."""
        return await self._run_text_micro_batch(
            items=payload.items,
            developer=developer,
            api_key_id=api_key_id,
            capability="sentiment",
            model_id=settings.MODEL_SENTIMENT_ID,
            response_cls=SentimentResponse,
        )

    # -------------------------------------------------------------------------
    # 2. Batch NER (Genuine Micro-Batched Inference)
    # -------------------------------------------------------------------------

    async def run_ner_batch(
        self,
        payload: BatchNERRequest,
        developer: Developer,
        api_key_id: Optional[uuid.UUID] = None,
    ) -> BatchResponse[NERResponse]:
        """Run batched Named Entity Recognition using micro-batched token classification."""
        return await self._run_text_micro_batch(
            items=payload.items,
            developer=developer,
            api_key_id=api_key_id,
            capability="ner",
            model_id=settings.MODEL_NER_ID,
            response_cls=NERResponse,
        )

    # -------------------------------------------------------------------------
    # 3. Batch Zero-Shot Classification (Grouped Batched NLI Inference)
    # -------------------------------------------------------------------------

    async def run_classification_batch(
        self,
        payload: BatchZeroShotRequest,
        developer: Developer,
        api_key_id: Optional[uuid.UUID] = None,
    ) -> BatchResponse[ZeroShotResponse]:
        """Run batched zero-shot classification grouped by (candidate_labels, multi_label)."""
        batch_start = time.perf_counter()
        model_id = settings.MODEL_ZERO_SHOT_ID
        results: List[Optional[BatchItemResult[ZeroShotResponse]]] = [None] * len(payload.items)
        grouped: Dict[Tuple[Tuple[str, ...], bool], List[Tuple[int, Optional[str], str]]] = defaultdict(list)
        clean_texts: Dict[int, str] = {}
        valid_count = 0

        for idx, item in enumerate(payload.items):
            try:
                labels = item.candidate_labels if item.candidate_labels is not None else payload.candidate_labels
                multi_label = item.multi_label if item.multi_label is not None else payload.multi_label
                clean_text, clean_labels = self.nlp_service.validate_zero_shot_inputs(item.text, labels)
                clean_texts[idx] = clean_text
                grouped[(tuple(clean_labels), bool(multi_label))].append((idx, item.id, clean_text))
                valid_count += 1
            except Exception as exc:
                results[idx] = BatchItemResult[ZeroShotResponse](
                    index=idx,
                    id=item.id,
                    status="error",
                    error=_map_exception_to_item_error(exc),
                )

        async def _execute() -> None:
            model = await self.registry.get_model(model_id)
            chunk_size = max(1, settings.BATCH_INFERENCE_CHUNK_SIZE)
            work_chunks: List[Tuple[List[str], bool, List[Tuple[int, Optional[str], str]]]] = []
            for (labels_tuple, multi_label), entries in grouped.items():
                labels_list = list(labels_tuple)
                for i in range(0, len(entries), chunk_size):
                    work_chunks.append((labels_list, multi_label, entries[i : i + chunk_size]))

            async def _process_group_chunk(
                work: Tuple[List[str], bool, List[Tuple[int, Optional[str], str]]]
            ) -> None:
                labels_list, multi_label, chunk = work
                texts = [entry[2] for entry in chunk]
                t0 = time.perf_counter()
                try:
                    outputs = await model.predict_batch(
                        {
                            "texts": texts,
                            "candidate_labels": labels_list,
                            "multi_label": multi_label,
                        },
                        batch_size=len(texts),
                    )
                    batch_ms = (time.perf_counter() - t0) * 1000
                    per_item_ms = round(batch_ms / len(chunk), 2)
                    for (idx, item_id, _), out in zip(chunk, outputs):
                        results[idx] = BatchItemResult[ZeroShotResponse](
                            index=idx,
                            id=item_id,
                            status="success",
                            cost_units=1,
                            latency_ms=per_item_ms,
                            result=ZeroShotResponse(**out, model_id=model_id, latency_ms=per_item_ms),
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if len(chunk) == 1:
                        idx, item_id, _ = chunk[0]
                        results[idx] = BatchItemResult[ZeroShotResponse](
                            index=idx,
                            id=item_id,
                            status="error",
                            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                            error=_map_exception_to_item_error(exc),
                        )
                        return
                    for idx, item_id, text in chunk:
                        item_t0 = time.perf_counter()
                        try:
                            out = await model.predict({
                                "text": text,
                                "candidate_labels": labels_list,
                                "multi_label": multi_label,
                            })
                            item_ms = round((time.perf_counter() - item_t0) * 1000, 2)
                            results[idx] = BatchItemResult[ZeroShotResponse](
                                index=idx,
                                id=item_id,
                                status="success",
                                cost_units=1,
                                latency_ms=item_ms,
                                result=ZeroShotResponse(**out, model_id=model_id, latency_ms=item_ms),
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as item_exc:
                            results[idx] = BatchItemResult[ZeroShotResponse](
                                index=idx,
                                id=item_id,
                                status="error",
                                latency_ms=round((time.perf_counter() - item_t0) * 1000, 2),
                                error=_map_exception_to_item_error(item_exc),
                            )

            await _run_bounded_workers(work_chunks, settings.BATCH_MAX_CONCURRENCY, _process_group_chunk)

        await self._run_with_quota_guard(
            developer=developer,
            api_key_id=api_key_id,
            capability="classification",
            valid_count=valid_count,
            clean_texts=clean_texts,
            results=results,
            execute_fn=_execute,
        )
        return self._finalize_response(
            "classification", "grouped_batched_inference", model_id, results, batch_start
        )

    # -------------------------------------------------------------------------
    # 4. Batch Keyword Extraction (Cross-Doc Deduplicated Batched Embeddings)
    # -------------------------------------------------------------------------

    async def run_keywords_batch(
        self,
        payload: BatchKeywordRequest,
        developer: Developer,
        api_key_id: Optional[uuid.UUID] = None,
    ) -> BatchResponse[KeywordResponse]:
        """Run keyword extraction batch with cross-document deduplicated embedding generation."""
        batch_start = time.perf_counter()
        model_id = settings.MODEL_EMBEDDING_ID
        results: List[Optional[BatchItemResult[KeywordResponse]]] = [None] * len(payload.items)
        clean_texts: Dict[int, str] = {}
        prepared_items: List[Tuple[int, Optional[str], str, int, List[str]]] = []
        valid_count = 0

        for idx, item in enumerate(payload.items):
            try:
                max_kw = item.max_keywords if item.max_keywords is not None else payload.max_keywords
                ngram_list = item.ngram_range if item.ngram_range is not None else payload.ngram_range
                cands_filter = item.candidates if item.candidates is not None else payload.candidates

                if len(ngram_list) != 2:
                    raise NLPValidationError(
                        "ngram_range must be a tuple or list of exactly two integers [min_n, max_n]."
                    )
                ngram_tuple = (ngram_list[0], ngram_list[1])
                clean_text = self.nlp_service.validate_keyword_inputs(
                    item.text, max_keywords=max_kw, ngram_range=ngram_tuple
                )
                candidate_phrases = extract_candidate_ngrams(
                    text=clean_text,
                    ngram_range=ngram_tuple,
                    candidate_filter=cands_filter,
                    max_candidates=settings.KEYWORD_MAX_CANDIDATES,
                )
                clean_texts[idx] = clean_text
                prepared_items.append((idx, item.id, clean_text, max_kw, candidate_phrases))
                valid_count += 1
            except Exception as exc:
                results[idx] = BatchItemResult[KeywordResponse](
                    index=idx,
                    id=item.id,
                    status="error",
                    error=_map_exception_to_item_error(exc),
                )

        async def _execute() -> None:
            import numpy as np

            items_needing_embeddings = [p for p in prepared_items if p[4]]
            # Fast-path stopword-only documents immediately
            for idx, item_id, _, _, cand_phrases in prepared_items:
                if not cand_phrases:
                    results[idx] = BatchItemResult[KeywordResponse](
                        index=idx,
                        id=item_id,
                        status="success",
                        cost_units=1,
                        latency_ms=0.0,
                        result=KeywordResponse(
                            keywords=[],
                            total_candidates=0,
                            model_id=model_id,
                            latency_ms=0.0,
                        ),
                    )

            if not items_needing_embeddings:
                return

            model = await self.registry.get_model(model_id)

            # Deduplicate all doc + candidate strings across the batch
            unique_texts: List[str] = []
            seen: set[str] = set()
            for _, _, clean_text, _, cand_phrases in items_needing_embeddings:
                for txt in [clean_text, *cand_phrases]:
                    if txt not in seen:
                        seen.add(txt)
                        unique_texts.append(txt)

            embed_chunk_size = max(8, settings.BATCH_INFERENCE_CHUNK_SIZE * 4)
            text_chunks = [
                unique_texts[i : i + embed_chunk_size]
                for i in range(0, len(unique_texts), embed_chunk_size)
            ]

            embedding_map: Dict[str, np.ndarray] = {}
            t_embed_start = time.perf_counter()
            fallback_items: List[Tuple[int, Optional[str], str, int, List[str]]] = []
            failed_texts: set[str] = set()

            async def _encode_chunk(chunk: List[str]) -> None:
                try:
                    vecs = await model.predict(chunk)
                    for txt, vec in zip(chunk, vecs):
                        embedding_map[txt] = vec
                except asyncio.CancelledError:
                    raise
                except Exception:
                    failed_texts.update(chunk)

            await _run_bounded_workers(text_chunks, settings.BATCH_MAX_CONCURRENCY, _encode_chunk)
            embed_elapsed_ms = (time.perf_counter() - t_embed_start) * 1000
            amortized_ms = round(embed_elapsed_ms / max(1, len(items_needing_embeddings)), 2)

            for entry in items_needing_embeddings:
                idx, item_id, clean_text, max_kw, cand_phrases = entry
                needed = [clean_text, *cand_phrases]
                if any(t in failed_texts or t not in embedding_map for t in needed):
                    fallback_items.append(entry)
                    continue
                doc_emb = embedding_map[clean_text]
                cand_embs = np.stack([embedding_map[c] for c in cand_phrases], axis=0)
                top_keywords = self.nlp_service.rank_keywords(
                    candidate_phrases=cand_phrases,
                    doc_emb=doc_emb,
                    cand_embs=cand_embs,
                    max_keywords=max_kw,
                )
                results[idx] = BatchItemResult[KeywordResponse](
                    index=idx,
                    id=item_id,
                    status="success",
                    cost_units=1,
                    latency_ms=amortized_ms,
                    result=KeywordResponse(
                        keywords=top_keywords,
                        total_candidates=len(cand_phrases),
                        model_id=model_id,
                        latency_ms=amortized_ms,
                    ),
                )

            # Isolated per-item fallback if any shared chunk failed
            for idx, item_id, clean_text, max_kw, cand_phrases in fallback_items:
                t0 = time.perf_counter()
                try:
                    vecs = await model.predict([clean_text, *cand_phrases])
                    top_keywords = self.nlp_service.rank_keywords(
                        candidate_phrases=cand_phrases,
                        doc_emb=vecs[0],
                        cand_embs=vecs[1:],
                        max_keywords=max_kw,
                    )
                    item_ms = round((time.perf_counter() - t0) * 1000, 2)
                    results[idx] = BatchItemResult[KeywordResponse](
                        index=idx,
                        id=item_id,
                        status="success",
                        cost_units=1,
                        latency_ms=item_ms,
                        result=KeywordResponse(
                            keywords=top_keywords,
                            total_candidates=len(cand_phrases),
                            model_id=model_id,
                            latency_ms=item_ms,
                        ),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    results[idx] = BatchItemResult[KeywordResponse](
                        index=idx,
                        id=item_id,
                        status="error",
                        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                        error=_map_exception_to_item_error(exc),
                    )

        await self._run_with_quota_guard(
            developer=developer,
            api_key_id=api_key_id,
            capability="keywords",
            valid_count=valid_count,
            clean_texts=clean_texts,
            results=results,
            execute_fn=_execute,
        )
        return self._finalize_response(
            "keywords", "deduplicated_batched_embeddings", model_id, results, batch_start
        )

    # -------------------------------------------------------------------------
    # 5. Batch Summarization (Bounded Concurrent Hierarchical Summarization)
    # -------------------------------------------------------------------------

    async def run_summarization_batch(
        self,
        payload: BatchSummarizationRequest,
        developer: Developer,
        api_key_id: Optional[uuid.UUID] = None,
    ) -> BatchResponse[SummarizationResponse]:
        """Run bounded concurrent summarization reusing chunking and hierarchical consolidation."""
        batch_start = time.perf_counter()
        model_id = settings.MODEL_SUMMARIZATION_ID
        results: List[Optional[BatchItemResult[SummarizationResponse]]] = [None] * len(payload.items)
        valid_entries: List[Tuple[int, Optional[str], str, int, int, bool, Optional[float]]] = []
        clean_texts: Dict[int, str] = {}

        for idx, item in enumerate(payload.items):
            try:
                min_len = item.min_length if item.min_length is not None else payload.min_length
                max_len = item.max_length if item.max_length is not None else payload.max_length
                do_sample = item.do_sample if item.do_sample is not None else payload.do_sample
                temp = item.temperature if item.temperature is not None else payload.temperature
                clean_text = self.nlp_service.validate_summarization_inputs(
                    text=item.text,
                    min_length=min_len,
                    max_length=max_len,
                    do_sample=do_sample,
                    temperature=temp,
                )
                clean_texts[idx] = clean_text
                valid_entries.append((idx, item.id, clean_text, min_len, max_len, do_sample, temp))
            except Exception as exc:
                results[idx] = BatchItemResult[SummarizationResponse](
                    index=idx,
                    id=item.id,
                    status="error",
                    error=_map_exception_to_item_error(exc),
                )

        async def _execute() -> None:
            model = await self.registry.get_model(model_id)

            async def _summarize_one(
                entry: Tuple[int, Optional[str], str, int, int, bool, Optional[float]]
            ) -> None:
                idx, item_id, clean_text, min_len, max_len, do_sample, temp = entry
                t0 = time.perf_counter()
                try:
                    summary_text, chunks_processed = await self.nlp_service.run_summarization_model(
                        model=model,
                        clean_text=clean_text,
                        min_length=min_len,
                        max_length=max_len,
                        do_sample=do_sample,
                        temperature=temp,
                    )
                    item_ms = round((time.perf_counter() - t0) * 1000, 2)
                    results[idx] = BatchItemResult[SummarizationResponse](
                        index=idx,
                        id=item_id,
                        status="success",
                        cost_units=chunks_processed,
                        latency_ms=item_ms,
                        result=SummarizationResponse(
                            summary=summary_text,
                            original_character_count=len(clean_text),
                            summary_character_count=len(summary_text),
                            chunks_processed=chunks_processed,
                            model_id=model_id,
                            latency_ms=item_ms,
                        ),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    results[idx] = BatchItemResult[SummarizationResponse](
                        index=idx,
                        id=item_id,
                        status="error",
                        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                        error=_map_exception_to_item_error(exc),
                    )

            await _run_bounded_workers(valid_entries, settings.BATCH_MAX_CONCURRENCY, _summarize_one)

        await self._run_with_quota_guard(
            developer=developer,
            api_key_id=api_key_id,
            capability="summarization",
            valid_count=len(valid_entries),
            clean_texts=clean_texts,
            results=results,
            execute_fn=_execute,
        )
        return self._finalize_response("summarization", "bounded_concurrent", model_id, results, batch_start)

    # -------------------------------------------------------------------------
    # 6. Batch Structured Extraction (Bounded Concurrent Local LLM Execution)
    # -------------------------------------------------------------------------

    async def run_extraction_batch(
        self,
        payload: BatchExtractionRequest,
        developer: Developer,
        api_key_id: Optional[uuid.UUID] = None,
    ) -> BatchResponse[ExtractionResponse]:
        """Run bounded concurrent Ollama structured extraction with per-item grounding and metering."""
        batch_start = time.perf_counter()
        default_model = payload.model or settings.OLLAMA_MODEL
        results: List[Optional[BatchItemResult[ExtractionResponse]]] = [None] * len(payload.items)
        valid_entries: List[Tuple[int, Optional[str], str, Dict[str, Any], str, str, float]] = []
        clean_texts: Dict[int, str] = {}

        for idx, item in enumerate(payload.items):
            try:
                schema_def = item.schema_definition if item.schema_definition is not None else payload.schema_definition
                fields = item.fields if item.fields is not None else payload.fields
                instructions = item.instructions if item.instructions is not None else payload.instructions
                item_model = item.model if item.model is not None else payload.model
                temp = item.temperature if item.temperature is not None else payload.temperature

                clean_text, target_schema, target_model, user_prompt, clamped_temp = (
                    self.extraction_service.validate_and_prepare(
                        text=item.text,
                        schema_definition=schema_def,
                        fields=fields,
                        instructions=instructions,
                        model=item_model,
                        temperature=temp,
                    )
                )
                clean_texts[idx] = clean_text
                valid_entries.append(
                    (idx, item.id, clean_text, target_schema, target_model, user_prompt, clamped_temp)
                )
            except Exception as exc:
                results[idx] = BatchItemResult[ExtractionResponse](
                    index=idx,
                    id=item.id,
                    status="error",
                    error=_map_exception_to_item_error(exc),
                )

        async def _execute() -> None:
            concurrency = max(1, min(settings.BATCH_MAX_CONCURRENCY, settings.BATCH_EXTRACTION_CONCURRENCY))

            async def _extract_one(
                entry: Tuple[int, Optional[str], str, Dict[str, Any], str, str, float]
            ) -> None:
                idx, item_id, clean_text, target_schema, target_model, user_prompt, clamped_temp = entry
                t0 = time.perf_counter()
                try:
                    ext_result = await self.extraction_service.execute_extraction_inference(
                        clean_text=clean_text,
                        target_schema=target_schema,
                        target_model=target_model,
                        user_prompt=user_prompt,
                        temperature=clamped_temp,
                    )
                    results[idx] = BatchItemResult[ExtractionResponse](
                        index=idx,
                        id=item_id,
                        status="success",
                        cost_units=ext_result["cost_units"],
                        latency_ms=ext_result["latency_ms"],
                        result=ExtractionResponse(**ext_result),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    results[idx] = BatchItemResult[ExtractionResponse](
                        index=idx,
                        id=item_id,
                        status="error",
                        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                        error=_map_exception_to_item_error(exc),
                    )

            await _run_bounded_workers(valid_entries, concurrency, _extract_one)

        await self._run_with_quota_guard(
            developer=developer,
            api_key_id=api_key_id,
            capability="extraction",
            valid_count=len(valid_entries),
            clean_texts=clean_texts,
            results=results,
            execute_fn=_execute,
        )
        return self._finalize_response("extraction", "bounded_concurrent", default_model, results, batch_start)
