"""Unit tests for BaseModel lifecycle contracts, concurrency, and inference offloading."""

import asyncio
import time
from typing import Any

import pytest

from src.core.exceptions import (
    ModelError,
    ModelInUseError,
    ModelInferenceError,
    ModelLoadError,
)
from src.core.models.base import BaseModel, ModelMetadata, ModelState


class DummyModel(BaseModel):
    """Test model implementation tracking lifecycle hooks and execution delay."""

    def __init__(self, metadata: ModelMetadata, delay: float = 0.0, fail_load: bool = False) -> None:
        super().__init__(metadata)
        self.delay = delay
        self.fail_load = fail_load
        self.weights_loaded = False

    def _load_weights(self, device: str) -> None:
        if self.fail_load:
            raise RuntimeError("Weight checkpoint file corrupted")
        time.sleep(self.delay)
        self.weights_loaded = True

    def _unload_weights(self) -> None:
        time.sleep(self.delay)
        self.weights_loaded = False

    def _execute_inference(self, inputs: Any, **kwargs: Any) -> Any:
        time.sleep(self.delay)
        if isinstance(inputs, str) and "CUDA OOM" in inputs:
            raise RuntimeError("CUDA out of memory: tried to allocate 2048 MiB")
        if isinstance(inputs, str) and "fail" in inputs:
            raise ValueError("Invalid inference token")
        return {"output": f"analyzed: {inputs}"}


@pytest.fixture
def sample_metadata() -> ModelMetadata:
    return ModelMetadata(
        model_id="test-bert",
        name="Test BERT Model",
        task="sentiment",
        version="1.0.0",
        vram_estimate_mb=500.0,
        ram_estimate_mb=300.0,
        max_concurrency=2,
        timeout_seconds=2.0,
    )


@pytest.mark.asyncio
async def test_model_lifecycle_transitions(sample_metadata: ModelMetadata):
    """Verify clean state transitions through UNLOADED -> READY -> UNLOADED."""
    model = DummyModel(sample_metadata)
    assert model.state == ModelState.UNLOADED
    assert not model.is_ready()
    assert model.device is None

    await model.load("cpu")
    assert model.state == ModelState.READY
    assert model.is_ready()
    assert model.device == "cpu"
    assert model.loaded_at is not None
    assert model.weights_loaded is True

    result = await model.predict("positive text")
    assert result == {"output": "analyzed: positive text"}
    assert model.total_inferences == 1

    await model.unload()
    assert model.state == ModelState.UNLOADED
    assert not model.is_ready()
    assert model.device is None
    assert model.weights_loaded is False


@pytest.mark.asyncio
async def test_model_load_failure(sample_metadata: ModelMetadata):
    """Verify that failure during _load_weights sets state to FAILED and raises ModelLoadError."""
    model = DummyModel(sample_metadata, fail_load=True)

    with pytest.raises(ModelLoadError) as exc_info:
        await model.load("cpu")

    assert model.state == ModelState.FAILED
    assert not model.is_ready()
    assert "corrupted" in str(exc_info.value.reason)
    assert model.error_message is not None


@pytest.mark.asyncio
async def test_model_predict_when_not_ready_fails(sample_metadata: ModelMetadata):
    """Verify predict on unloaded model raises ModelError."""
    model = DummyModel(sample_metadata)

    with pytest.raises(ModelError) as exc_info:
        await model.predict("some input")

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_model_concurrency_semaphore(sample_metadata: ModelMetadata):
    """Verify that model concurrency is bounded by max_concurrency."""
    # max_concurrency is 2, set delay to 0.1s
    model = DummyModel(sample_metadata, delay=0.1)
    await model.load("cpu")

    max_concurrent_seen = 0

    async def run_task(idx: int):
        nonlocal max_concurrent_seen
        # During inference, check active_inferences
        cur = model.active_inferences
        if cur > max_concurrent_seen:
            max_concurrent_seen = cur
        return await model.predict(f"text {idx}")

    # Launch 5 concurrent inference tasks
    results = await asyncio.gather(*(run_task(i) for i in range(5)))
    assert len(results) == 5
    # Max active should never exceed max_concurrency (2)
    assert max_concurrent_seen <= 2
    assert model.total_inferences == 5
    assert model.active_inferences == 0

    await model.unload()


@pytest.mark.asyncio
async def test_model_unload_while_busy_fails(sample_metadata: ModelMetadata):
    """Verify unload() raises ModelInUseError if inference is currently active."""
    model = DummyModel(sample_metadata, delay=0.2)
    await model.load("cpu")

    async def long_running_task():
        return await model.predict("long running input")

    task = asyncio.create_task(long_running_task())
    await asyncio.sleep(0.05)  # Ensure task is running

    # Attempt to unload while active
    with pytest.raises(ModelInUseError) as exc_info:
        await model.unload()

    assert exc_info.value.status_code == 409
    assert exc_info.value.model_id == "test-bert"

    # Await completion
    await task
    # Now unload should succeed
    await model.unload()
    assert model.state == ModelState.UNLOADED


@pytest.mark.asyncio
async def test_model_inference_timeout(sample_metadata: ModelMetadata):
    """Verify inference timing out raises ModelInferenceError."""
    sample_metadata.timeout_seconds = 0.05
    model = DummyModel(sample_metadata, delay=0.2)
    await model.load("cpu")

    with pytest.raises(ModelInferenceError) as exc_info:
        await model.predict("slow input")

    assert "timed out" in str(exc_info.value.reason)
    assert model.active_inferences == 0  # Cleaned up in finally

    await model.unload()


@pytest.mark.asyncio
async def test_model_cuda_oom_handling_during_inference(sample_metadata: ModelMetadata):
    """Verify that CUDA OOM during inference triggers cleanup and raises ModelInferenceError."""
    model = DummyModel(sample_metadata)
    await model.load("cpu")

    with pytest.raises(ModelInferenceError) as exc_info:
        await model.predict("CUDA OOM text")

    assert "out of memory" in str(exc_info.value.reason).lower()
    assert model.active_inferences == 0

    await model.unload()
