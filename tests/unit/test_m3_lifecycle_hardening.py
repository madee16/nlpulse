"""Regression tests for M3 model lifecycle hardening: thread safety, timeout, and CUDA OOM."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from src.core.device import DeviceManager
from src.core.exceptions import (
    ModelInUseError,
    ModelInferenceError,
)
from src.core.models.base import BaseModel, ModelMetadata, ModelState


class ControlledBlockingModel(BaseModel):
    """Model that blocks in a worker thread until explicitly unblocked or for a fixed duration."""

    def __init__(self, metadata: ModelMetadata) -> None:
        super().__init__(metadata)
        self.blocking_duration = 0.0
        self.unblock_event = None

    def _load_weights(self, device: str) -> None:
        pass

    def _unload_weights(self) -> None:
        pass

    def _execute_inference(self, inputs: str, **kwargs):
        if self.blocking_duration > 0:
            time.sleep(self.blocking_duration)
        if self.unblock_event:
            self.unblock_event.wait(timeout=2.0)
        if "oom" in inputs.lower():
            raise RuntimeError("CUDA out of memory: Tried to allocate 4.00 GiB")
        return {"result": f"processed: {inputs}"}


@pytest.fixture
def blocking_metadata() -> ModelMetadata:
    return ModelMetadata(
        model_id="controlled-blocking",
        name="Controlled Blocking Model",
        task="test",
        max_concurrency=1,
        timeout_seconds=0.08,
    )


@pytest.mark.asyncio
async def test_timeout_preserves_active_inference_until_thread_completes(blocking_metadata: ModelMetadata):
    """Verify that when an inference times out:
    1. The caller immediately receives a timeout error.
    2. The model remains accounted for (active_inferences > 0) while the thread is still running.
    3. Model cannot be unloaded during this time (raises ModelInUseError).
    4. Once the worker thread finishes, active_inferences decrements to 0 and unload succeeds.
    """
    model = ControlledBlockingModel(blocking_metadata)
    model.blocking_duration = 0.25  # Thread runs for 250ms, timeout is 80ms
    await model.load("cpu")

    # 1. Dispatch inference that will timeout on the caller side
    start = time.perf_counter()
    with pytest.raises(ModelInferenceError) as exc_info:
        await model.predict("test slow input")
    elapsed = time.perf_counter() - start

    # Caller timed out around 80ms, well before the 250ms thread finishes
    assert elapsed < 0.20
    assert "timed out" in str(exc_info.value.reason)

    # 2. Crucial invariant: The worker thread is STILL running, so active_inferences must remain 1
    assert model.active_inferences == 1
    assert model.is_busy() is True

    # 3. Model unload MUST be rejected while the worker thread is still executing
    with pytest.raises(ModelInUseError) as unload_err:
        await model.unload()
    assert unload_err.value.status_code == 409

    # 4. Wait for the background worker thread to finish its 250ms execution
    await asyncio.sleep(0.25)

    # After the thread completes, active_inferences is properly cleared
    assert model.active_inferences == 0
    assert model.is_busy() is False

    # Now unload cleanly succeeds
    await model.unload()
    assert model.state == ModelState.UNLOADED


@pytest.mark.asyncio
async def test_cuda_oom_distinguishes_live_tensors_and_flushes_cache():
    """Verify handle_cuda_oom inspects allocated vs reserved memory and releases unused cache."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1
    # 2 GB active live tensors, 4 GB total reserved by allocator -> 2 GB reclaimable unused cache
    mock_torch.cuda.memory_allocated.return_value = 2 * 1024 * 1024 * 1024
    mock_torch.cuda.memory_reserved.return_value = 4 * 1024 * 1024 * 1024

    with patch.dict("sys.modules", {"torch": mock_torch}):
        dm = DeviceManager(preference="cpu")
        with patch.object(dm, "is_cuda_available", return_value=True):
            diagnostics = dm.handle_cuda_oom()

            assert diagnostics["live_allocated_mb"] == 2048.0
            assert diagnostics["reserved_mb"] == 4096.0
            assert diagnostics["reclaimable_cached_mb"] == 2048.0
            assert diagnostics["cache_flushed"] is True
            mock_torch.cuda.empty_cache.assert_called_once()


@pytest.mark.asyncio
async def test_cuda_oom_during_inference_preserves_model_ready_state(blocking_metadata: ModelMetadata):
    """Verify that when an inference call throws CUDA OOM, the model state is not corrupted."""
    blocking_metadata.timeout_seconds = 2.0
    model = ControlledBlockingModel(blocking_metadata)
    await model.load("cpu")

    with patch("src.core.models.base.device_manager.handle_cuda_oom") as mock_oom_handler:
        with pytest.raises(ModelInferenceError) as exc_info:
            await model.predict("cause cuda OOM")

        assert "out of memory" in str(exc_info.value.reason).lower()
        mock_oom_handler.assert_called_once()

    # Model remains READY for future requests; state is not corrupted
    assert model.state == ModelState.READY
    assert model.active_inferences == 0

    await model.unload()
