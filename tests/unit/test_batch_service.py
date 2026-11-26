"""Unit tests for M7 BatchService orchestration, partial failures, batching, concurrency, cancellation, and metering."""

import asyncio
import time
from typing import Any, List
import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.core.device import DeviceManager
from src.core.exceptions import (
    OllamaTimeoutError,
    OllamaUnavailableError,
    QuotaExceededError,
)
from src.core.models.base import ModelMetadata, ModelState
from src.core.models.nlp import (
    EmbeddingModel,
    NERModel,
    SentimentModel,
    SummarizationModel,
    ZeroShotClassificationModel,
)
from src.models.usage_record import UsageRecord
from src.schemas.batch import (
    BatchExtractionRequest,
    BatchKeywordRequest,
    BatchNERRequest,
    BatchSentimentRequest,
    BatchSummarizationRequest,
    BatchZeroShotRequest,
)
from src.services.batch_service import BatchService
from src.services.developer_service import DeveloperService
from src.services.model_registry import ModelRegistry
from src.services.ollama_client import OllamaClient
from src.services.usage_service import quota_reservation_manager

settings = get_settings()


@pytest.fixture(autouse=True)
async def reset_quota_reservations():
    await quota_reservation_manager.reset()
    yield
    await quota_reservation_manager.reset()


def _make_registry():
    """Build a test ModelRegistry with batch-observing fake HF pipelines."""
    reg = ModelRegistry(device_mgr=DeviceManager(preference="cpu"), max_cached_models=5)
    tracker: dict[str, Any] = {
        "sentiment_calls": [],
        "ner_calls": [],
        "zero_shot_calls": [],
        "embedding_calls": [],
        "sentiment_loads": 0,
        "sum_active": 0,
        "sum_max_active": 0,
    }

    # 1. Sentiment
    class FakeHFSentiment:
        tokenizer = object()
        model = object()

        def __call__(self, inputs, **kwargs):
            tracker["sentiment_calls"].append(inputs)
            if isinstance(inputs, list):
                out = []
                for t in inputs:
                    if "poison" in t.lower():
                        raise RuntimeError("Simulated poison input crash")
                    if "great" in t.lower():
                        out.append([{"label": "POSITIVE", "score": 0.98}, {"label": "NEGATIVE", "score": 0.02}])
                    else:
                        out.append([{"label": "NEGATIVE", "score": 0.91}, {"label": "POSITIVE", "score": 0.09}])
                return out
            if "poison" in inputs.lower():
                raise RuntimeError("Simulated single poison crash")
            if "great" in inputs.lower():
                return [{"label": "POSITIVE", "score": 0.98}, {"label": "NEGATIVE", "score": 0.02}]
            return [{"label": "NEGATIVE", "score": 0.91}, {"label": "POSITIVE", "score": 0.09}]

    def sent_factory():
        tracker["sentiment_loads"] += 1
        return FakeHFSentiment()

    s_meta = ModelMetadata(model_id=settings.MODEL_SENTIMENT_ID, name="Sent", task="sentiment")
    reg.register(s_meta, lambda: SentimentModel(s_meta, pipeline_factory=sent_factory))

    # 2. NER
    class FakeHFNER:
        tokenizer = object()
        model = object()

        def __call__(self, inputs, **kwargs):
            tracker["ner_calls"].append(inputs)
            if isinstance(inputs, list):
                return [
                    [{"entity_group": "PER", "score": 0.99, "word": "Alice", "start": 0, "end": 5}]
                    if "Alice" in t
                    else []
                    for t in inputs
                ]
            return [{"entity_group": "PER", "score": 0.99, "word": "Alice", "start": 0, "end": 5}] if "Alice" in inputs else []

    n_meta = ModelMetadata(model_id=settings.MODEL_NER_ID, name="NER", task="ner")
    reg.register(n_meta, lambda: NERModel(n_meta, pipeline_factory=lambda: FakeHFNER()))

    # 3. Zero-Shot
    class FakeHFZeroShot:
        tokenizer = object()
        model = object()

        def __call__(self, inputs, candidate_labels, multi_label=False, **kwargs):
            tracker["zero_shot_calls"].append((inputs, list(candidate_labels), multi_label))
            if isinstance(inputs, list):
                return [
                    {"sequence": seq, "labels": list(candidate_labels), "scores": [0.9 / (i + 1) for i in range(len(candidate_labels))]}
                    for seq in inputs
                ]
            return {"sequence": inputs, "labels": list(candidate_labels), "scores": [0.9 / (i + 1) for i in range(len(candidate_labels))]}

    z_meta = ModelMetadata(model_id=settings.MODEL_ZERO_SHOT_ID, name="ZS", task="zero-shot")
    reg.register(z_meta, lambda: ZeroShotClassificationModel(z_meta, pipeline_factory=lambda: FakeHFZeroShot()))

    # 4. Embedding
    def fake_embed(texts: List[str]):
        tracker["embedding_calls"].append(list(texts))
        vecs = []
        for t in texts:
            if "fastapi" in t.lower() or "api" in t.lower():
                v = np.array([0.9, 0.1, 0.0], dtype=np.float32)
            else:
                v = np.array([0.3, 0.3, 0.3], dtype=np.float32)
            vecs.append(v / np.linalg.norm(v))
        return np.array(vecs, dtype=np.float32)

    e_meta = ModelMetadata(model_id=settings.MODEL_EMBEDDING_ID, name="Emb", task="embedding")
    reg.register(e_meta, lambda: EmbeddingModel(e_meta, pipeline_factory=lambda: fake_embed))

    # 5. Summarization
    def fake_sum(text: str, **kwargs):
        tracker["sum_active"] += 1
        tracker["sum_max_active"] = max(tracker["sum_max_active"], tracker["sum_active"])
        time.sleep(0.03)
        tracker["sum_active"] -= 1
        return [{"summary_text": f"Summary of {text[:25]}"}]

    sum_meta = ModelMetadata(
        model_id=settings.MODEL_SUMMARIZATION_ID,
        name="Sum",
        task="summarization",
        max_concurrency=4,
    )
    reg.register(sum_meta, lambda: SummarizationModel(sum_meta, pipeline_factory=lambda: fake_sum))

    return reg, tracker


@pytest.mark.asyncio
async def test_batch_sentiment_micro_batching_model_reuse_and_metering(db_session: AsyncSession, monkeypatch):
    """Verify sentiment uses micro-batched tensor calls, loads model once, preserves order/IDs, and meters accurately."""
    monkeypatch.setattr(settings, "BATCH_INFERENCE_CHUNK_SIZE", 3)
    dev = await DeveloperService(db_session).create_developer(email="b_sent@example.com", name="Batch Sent")
    reg, tracker = _make_registry()
    svc = BatchService(session=db_session, registry=reg)

    req = BatchSentimentRequest(
        items=[
            {"id": "a", "text": "Great product!"},
            {"id": "b", "text": "Bad outage."},
            {"id": "c", "text": "Great speed!"},
            {"id": "d", "text": "Awful latency."},
            {"id": "e", "text": "Great docs!"},
        ]
    )
    resp = await svc.run_sentiment_batch(req, developer=dev)

    assert resp.capability == "sentiment"
    assert resp.execution_mode == "batched_inference"
    assert resp.total_items == 5
    assert resp.succeeded_count == 5
    assert resp.failed_count == 0
    assert resp.total_cost_units == 5
    assert [r.id for r in resp.results] == ["a", "b", "c", "d", "e"]
    assert [r.index for r in resp.results] == [0, 1, 2, 3, 4]
    assert [r.result.label for r in resp.results] == ["POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE"]

    # Loaded once and dispatched in 2 micro-batches (3 + 2) instead of 5 separate calls
    assert tracker["sentiment_loads"] == 1
    assert len(tracker["sentiment_calls"]) == 2
    assert len(tracker["sentiment_calls"][0]) == 3
    assert len(tracker["sentiment_calls"][1]) == 2

    rows = (await db_session.execute(select(UsageRecord).where(UsageRecord.developer_id == dev.id))).scalars().all()
    assert len(rows) == 5
    assert all(r.endpoint == f"{settings.API_V1_PREFIX}/nlp/batch/sentiment" for r in rows)
    assert sum(r.cost_units for r in rows) == resp.total_cost_units


@pytest.mark.asyncio
async def test_batch_sentiment_partial_validation_and_poison_isolation(db_session: AsyncSession):
    """Verify empty text and poison item fail individually without invalidating healthy siblings."""
    dev = await DeveloperService(db_session).create_developer(email="b_poison@example.com", name="Poison")
    reg, _ = _make_registry()
    svc = BatchService(session=db_session, registry=reg)

    req = BatchSentimentRequest(
        items=[
            {"id": "ok-1", "text": "Great reliability!"},
            {"id": "empty-1", "text": "   "},
            {"id": "poison-1", "text": "This poison text crashes batch forward pass"},
            {"id": "ok-2", "text": "Terrible bug."},
        ]
    )
    resp = await svc.run_sentiment_batch(req, developer=dev)

    assert resp.total_items == 4
    assert resp.succeeded_count == 2
    assert resp.failed_count == 2
    assert resp.total_cost_units == 2

    assert resp.results[0].status == "success"
    assert resp.results[0].result.label == "POSITIVE"

    assert resp.results[1].status == "error"
    assert resp.results[1].cost_units == 0
    assert resp.results[1].error.status_code == 422
    assert resp.results[1].error.code == "validation_error"

    assert resp.results[2].status == "error"
    assert resp.results[2].cost_units == 0
    assert resp.results[2].error.status_code == 500
    assert resp.results[2].error.code == "model_inference_error"

    assert resp.results[3].status == "success"
    assert resp.results[3].result.label == "NEGATIVE"

    # Only 2 succeeded items were metered in DB
    rows = (await db_session.execute(select(UsageRecord).where(UsageRecord.developer_id == dev.id))).scalars().all()
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_batch_ner_single_and_multi_item(db_session: AsyncSession):
    """Verify NER batch works for single-item and multi-item batches with batched execution."""
    dev = await DeveloperService(db_session).create_developer(email="b_ner@example.com", name="NER")
    reg, tracker = _make_registry()
    svc = BatchService(session=db_session, registry=reg)

    single_resp = await svc.run_ner_batch(
        BatchNERRequest(items=[{"id": "s1", "text": "Alice visited Berlin."}]),
        developer=dev,
    )
    assert single_resp.succeeded_count == 1
    assert single_resp.results[0].result.entity_count == 1

    multi_resp = await svc.run_ner_batch(
        BatchNERRequest(
            items=[
                {"id": "m1", "text": "Alice works here."},
                {"id": "m2", "text": ""},
                {"id": "m3", "text": "No names at all."},
            ]
        ),
        developer=dev,
    )
    assert multi_resp.execution_mode == "batched_inference"
    assert multi_resp.succeeded_count == 2
    assert multi_resp.failed_count == 1
    assert multi_resp.results[0].result.entity_count == 1
    assert multi_resp.results[1].status == "error"
    assert multi_resp.results[2].result.entity_count == 0
    # Valid items in multi-item batch went through 1 batched call
    assert len(tracker["ner_calls"][-1]) == 2


@pytest.mark.asyncio
async def test_batch_zero_shot_grouped_batching_and_partial_label_errors(db_session: AsyncSession):
    """Verify zero-shot groups shared label specs for batched inference and isolates bad label items."""
    dev = await DeveloperService(db_session).create_developer(email="b_zs@example.com", name="ZS")
    reg, tracker = _make_registry()
    svc = BatchService(session=db_session, registry=reg)

    req = BatchZeroShotRequest(
        candidate_labels=["tech", "finance"],
        items=[
            {"id": "g1-a", "text": "FastAPI release"},
            {"id": "bad", "text": "Missing labels", "candidate_labels": []},
            {"id": "g1-b", "text": "Cloud earnings"},
            {"id": "g2", "text": "Sports finals", "candidate_labels": ["sports", "news"], "multi_label": True},
        ],
    )
    resp = await svc.run_classification_batch(req, developer=dev)

    assert resp.execution_mode == "grouped_batched_inference"
    assert resp.succeeded_count == 3
    assert resp.failed_count == 1
    assert resp.results[0].result.top_label == "tech"
    assert resp.results[1].status == "error"
    assert resp.results[2].result.top_label == "tech"
    assert resp.results[3].result.top_label == "sports"
    assert resp.results[3].result.multi_label is True

    # Two distinct label groups -> 2 batched pipeline calls (group1 had 2 texts, group2 had 1)
    assert len(tracker["zero_shot_calls"]) == 2
    assert tracker["zero_shot_calls"][0][0] == ["FastAPI release", "Cloud earnings"]


@pytest.mark.asyncio
async def test_batch_keywords_deduplicated_embedding_batch_and_stopword_item(db_session: AsyncSession):
    """Verify keyword batch deduplicates shared candidate strings into batched embedding calls."""
    dev = await DeveloperService(db_session).create_developer(email="b_kw@example.com", name="KW")
    reg, tracker = _make_registry()
    svc = BatchService(session=db_session, registry=reg)

    req = BatchKeywordRequest(
        max_keywords=2,
        items=[
            {"id": "k1", "text": "FastAPI builds fast APIs in Python."},
            {"id": "k2", "text": "FastAPI and Python power APIs."},
            {"id": "k3-stop", "text": "To be or not to be that is the."},
            {"id": "k4-bad", "text": "Valid text", "ngram_range": [3, 1]},
        ],
    )
    resp = await svc.run_keywords_batch(req, developer=dev)

    assert resp.execution_mode == "deduplicated_batched_embeddings"
    assert resp.succeeded_count == 3
    assert resp.failed_count == 1
    assert len(resp.results[0].result.keywords) == 2
    assert resp.results[2].result.keywords == []
    assert resp.results[2].result.total_candidates == 0
    assert resp.results[3].status == "error"

    # Embedded all unique texts across k1 & k2 in ONE deduplicated chunk call
    assert len(tracker["embedding_calls"]) == 1
    embedded_batch = tracker["embedding_calls"][0]
    assert len(embedded_batch) == len(set(embedded_batch))


@pytest.mark.asyncio
async def test_batch_summarization_bounded_concurrency_and_chunk_cost_units(db_session: AsyncSession, monkeypatch):
    """Verify summarization runs bounded concurrent workers and sums per-item chunk cost_units."""
    monkeypatch.setattr(settings, "BATCH_MAX_CONCURRENCY", 2)
    monkeypatch.setattr(settings, "SUMMARIZATION_CHUNK_SIZE", 100)
    dev = await DeveloperService(db_session).create_developer(email="b_sum@example.com", name="Sum")
    reg, tracker = _make_registry()
    svc = BatchService(session=db_session, registry=reg)

    long_doc = " ".join(f"Sentence {i} explains distributed vector indexing clearly." for i in range(6))
    req = BatchSummarizationRequest(
        min_length=10,
        max_length=50,
        items=[
            {"id": "s1", "text": "Short document about FastAPI."},
            {"id": "s2-bad", "text": "Valid text", "min_length": 100, "max_length": 20},
            {"id": "s3-long", "text": long_doc},
            {"id": "s4", "text": "Another concise paragraph."},
            {"id": "s5", "text": "Final short paragraph."},
        ],
    )
    resp = await svc.run_summarization_batch(req, developer=dev)

    assert resp.execution_mode == "bounded_concurrent"
    assert resp.succeeded_count == 4
    assert resp.failed_count == 1
    assert tracker["sum_max_active"] <= 2
    assert resp.results[2].result.chunks_processed > 1
    assert resp.results[2].cost_units == resp.results[2].result.chunks_processed
    assert resp.total_cost_units == sum(r.cost_units for r in resp.results if r.status == "success")


@pytest.mark.asyncio
async def test_batch_extraction_bounded_concurrency_and_partial_ollama_failures(db_session: AsyncSession, monkeypatch):
    """Verify extraction batch handles mixed success, schema validation errors, and Ollama timeout/unavailable errors."""
    monkeypatch.setattr(settings, "BATCH_EXTRACTION_CONCURRENCY", 2)
    dev = await DeveloperService(db_session).create_developer(email="b_ext@example.com", name="Ext")

    active = 0
    max_active = 0

    async def fake_generate(prompt: str, **kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        if "TIMEOUT_DOC" in prompt:
            raise OllamaTimeoutError("Timed out")
        if "BAD_JSON_DOC" in prompt:
            return {
                "response": "Not valid JSON at all",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "total_duration_ns": 100,
                "model": "qwen2.5:3b",
            }
        return {
            "response": '{"company": "Acme"}',
            "prompt_tokens": 100,
            "completion_tokens": 100,
            "total_tokens": 200,
            "total_duration_ns": 100,
            "model": "qwen2.5:3b",
        }

    mock_ollama = OllamaClient()
    mock_ollama.generate = fake_generate  # type: ignore[method-assign]
    reg, _ = _make_registry()
    svc = BatchService(session=db_session, registry=reg, ollama_client=mock_ollama)

    req = BatchExtractionRequest(
        fields=[{"name": "company", "type": "string"}],
        items=[
            {"id": "e1", "text": "Acme sells software."},
            {"id": "e2-timeout", "text": "TIMEOUT_DOC details"},
            {"id": "e3-badjson", "text": "BAD_JSON_DOC details"},
            {"id": "e4-noschema", "text": "Missing schema doc", "fields": []},
            {"id": "e5", "text": "Acme opened a lab."},
        ],
    )
    # Note: e4-noschema inherits batch-level fields when fields=[], wait: empty list [] explicitly overrides!
    resp = await svc.run_extraction_batch(req, developer=dev)

    assert resp.execution_mode == "bounded_concurrent"
    assert max_active <= 2
    assert resp.succeeded_count == 2
    assert resp.failed_count == 3
    assert resp.results[0].status == "success"
    assert resp.results[0].result.extracted_data == {"company": "Acme"}
    assert resp.results[1].error.status_code == 504
    assert resp.results[1].error.code == "ollama_timeout"
    assert resp.results[2].error.status_code == 422
    assert resp.results[3].error.status_code == 422
    assert resp.results[4].status == "success"

    rows = (await db_session.execute(select(UsageRecord).where(UsageRecord.developer_id == dev.id))).scalars().all()
    assert len(rows) == 2
    assert sum(r.cost_units for r in rows) == resp.total_cost_units


@pytest.mark.asyncio
async def test_batch_cancellation_releases_quota_and_preserves_m3_lifecycle(db_session: AsyncSession):
    """Verify cancelling an in-flight batch releases quota, bills 0 usage, and keeps M3 model tracking clean."""
    dev = await DeveloperService(db_session).create_developer(email="b_cancel@example.com", name="Cancel")
    dev.monthly_request_quota = 5
    await db_session.commit()

    reg = ModelRegistry(device_mgr=DeviceManager(preference="cpu"))

    loop = asyncio.get_running_loop()
    inference_started = asyncio.Event()

    def slow_sum(text: str, **kwargs):
        loop.call_soon_threadsafe(inference_started.set)
        time.sleep(0.15)
        return [{"summary_text": "Delayed"}]

    sum_meta = ModelMetadata(model_id=settings.MODEL_SUMMARIZATION_ID, name="SlowSum", task="summarization")
    reg.register(sum_meta, lambda: SummarizationModel(sum_meta, pipeline_factory=lambda: slow_sum))

    svc = BatchService(session=db_session, registry=reg)
    req = BatchSummarizationRequest(
        items=[{"text": f"Document {i} for cancellation test."} for i in range(4)]
    )

    task = asyncio.create_task(svc.run_summarization_batch(req, developer=dev))
    await asyncio.wait_for(inference_started.wait(), timeout=2.0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    # Quota reservation must be released immediately
    assert quota_reservation_manager.get_reserved(dev.id) == 0
    rows = (await db_session.execute(select(UsageRecord).where(UsageRecord.developer_id == dev.id))).scalars().all()
    assert len(rows) == 0

    # Wait for shielded worker thread to finish, then model can cleanly unload
    await asyncio.sleep(0.2)
    model = await reg.get_model(settings.MODEL_SUMMARIZATION_ID)
    assert model.active_inferences == 0
    await reg.unload_model(settings.MODEL_SUMMARIZATION_ID)
    assert not reg.is_loaded(settings.MODEL_SUMMARIZATION_ID)


@pytest.mark.asyncio
async def test_batch_all_invalid_items_never_loads_model_or_charges(db_session: AsyncSession):
    """Verify a batch where every item fails pre-validation skips model load and usage charging."""
    dev = await DeveloperService(db_session).create_developer(email="b_allbad@example.com", name="AllBad")
    reg, tracker = _make_registry()
    svc = BatchService(session=db_session, registry=reg)

    resp = await svc.run_sentiment_batch(
        BatchSentimentRequest(items=[{"text": ""}, {"text": "   "}]),
        developer=dev,
    )
    assert resp.succeeded_count == 0
    assert resp.failed_count == 2
    assert resp.total_cost_units == 0
    assert tracker["sentiment_loads"] == 0
    assert not reg.is_loaded(settings.MODEL_SENTIMENT_ID)
    rows = (await db_session.execute(select(UsageRecord).where(UsageRecord.developer_id == dev.id))).scalars().all()
    assert len(rows) == 0


@pytest.mark.asyncio
async def test_batch_insufficient_quota_raises_429_before_inference(db_session: AsyncSession):
    """Verify insufficient remaining quota rejects batch before executing model inference."""
    dev = await DeveloperService(db_session).create_developer(email="b_lowq@example.com", name="LowQuota")
    dev.monthly_request_quota = 2
    await db_session.commit()

    reg, tracker = _make_registry()
    svc = BatchService(session=db_session, registry=reg)

    with pytest.raises(QuotaExceededError):
        await svc.run_sentiment_batch(
            BatchSentimentRequest(items=[{"text": "one"}, {"text": "two"}, {"text": "three"}]),
            developer=dev,
        )

    assert tracker["sentiment_loads"] == 0
    assert quota_reservation_manager.get_reserved(dev.id) == 0
