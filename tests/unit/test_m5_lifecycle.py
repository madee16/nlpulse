"""Unit tests for M5 model lifecycle, lazy loading, LRU eviction, and concurrency."""

import asyncio
import numpy as np
import pytest

from src.core.device import DeviceManager
from src.core.models.base import ModelMetadata, ModelState
from src.core.models.nlp import EmbeddingModel, SummarizationModel
from src.services.model_registry import ModelRegistry


@pytest.mark.asyncio
async def test_m5_models_lazy_loading_and_reuse():
    """Verify EmbeddingModel and SummarizationModel remain uninitialized until requested, then reuse instance."""
    device_mgr = DeviceManager(preference="cpu")
    registry = ModelRegistry(device_mgr=device_mgr, max_cached_models=3)

    emb_load_count = 0
    def emb_factory():
        nonlocal emb_load_count
        emb_load_count += 1
        return lambda texts: np.ones((len(texts), 384), dtype=np.float32)

    emb_meta = ModelMetadata(model_id="lazy-emb", name="Lazy Embedding", task="embedding")
    registry.register(emb_meta, lambda: EmbeddingModel(emb_meta, pipeline_factory=emb_factory))

    # Not loaded yet
    assert registry.is_loaded("lazy-emb") is False
    assert emb_load_count == 0

    # First access loads model
    model1 = await registry.get_model("lazy-emb")
    assert registry.is_loaded("lazy-emb") is True
    assert model1.state == ModelState.READY
    assert emb_load_count == 1

    # Second access reuses loaded model
    model2 = await registry.get_model("lazy-emb")
    assert model1 is model2
    assert emb_load_count == 1


@pytest.mark.asyncio
async def test_m5_model_concurrency_semaphore():
    """Verify per-model concurrency limit prevents overwhelming compute hardware."""
    meta = ModelMetadata(
        model_id="conc-sum",
        name="Concurrent Summarizer",
        task="summarization",
        max_concurrency=1,
    )

    concurrent_count = 0
    max_observed_concurrency = 0

    def slow_pipe(text: str, **kwargs):
        nonlocal concurrent_count, max_observed_concurrency
        concurrent_count += 1
        max_observed_concurrency = max(max_observed_concurrency, concurrent_count)
        import time
        time.sleep(0.05)
        concurrent_count -= 1
        return [{"summary_text": "Slow summary"}]

    model = SummarizationModel(meta, pipeline_factory=lambda: slow_pipe)
    await model.load("cpu")

    # Launch 3 parallel requests
    tasks = [
        model.predict({"text": f"Document {i} to summarize"})
        for i in range(3)
    ]
    results = await asyncio.gather(*tasks)

    assert len(results) == 3
    # Max concurrency was constrained to 1
    assert max_observed_concurrency == 1
    await model.unload()
    assert model.state == ModelState.UNLOADED


@pytest.mark.asyncio
async def test_m5_lru_cache_eviction_with_vram_limits():
    """Verify LRU cache limits total simultaneously loaded models, evicting the least recently used idle model."""
    device_mgr = DeviceManager(preference="cpu")
    # Set small cache limit of 2 models
    registry = ModelRegistry(device_mgr=device_mgr, max_cached_models=2)

    meta_a = ModelMetadata(model_id="m-emb-1", name="Emb 1", task="embedding")
    meta_b = ModelMetadata(model_id="m-emb-2", name="Emb 2", task="embedding")
    meta_c = ModelMetadata(model_id="m-sum-1", name="Sum 1", task="summarization")

    registry.register(meta_a, lambda: EmbeddingModel(meta_a, pipeline_factory=lambda: lambda t: np.zeros((len(t), 384))))
    registry.register(meta_b, lambda: EmbeddingModel(meta_b, pipeline_factory=lambda: lambda t: np.zeros((len(t), 384))))
    registry.register(meta_c, lambda: SummarizationModel(meta_c, pipeline_factory=lambda: lambda t, **kw: [{"summary_text": "s"}]))

    # Load A and B
    m1 = await registry.get_model("m-emb-1")
    await asyncio.sleep(0.01)
    await registry.get_model("m-emb-2")
    await asyncio.sleep(0.01)
    assert registry.is_loaded("m-emb-1") is True
    assert registry.is_loaded("m-emb-2") is True
    assert registry.get_loaded_count() == 2

    # Touch A with an inference to update its last_used_at timestamp
    await m1.predict(["test"])
    await asyncio.sleep(0.01)

    # Load C -> triggers eviction of B (least recently used)
    await registry.get_model("m-sum-1")
    assert registry.get_loaded_count() == 2
    assert registry.is_loaded("m-emb-1") is True
    assert registry.is_loaded("m-sum-1") is True
    assert registry.is_loaded("m-emb-2") is False  # Evicted!
