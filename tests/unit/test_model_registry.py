"""Unit tests for ModelRegistry lazy loading, concurrency locks, and LRU eviction."""

import asyncio
import time
from typing import Any, List

import pytest

from src.core.device import DeviceManager
from src.core.exceptions import (
    ModelInUseError,
    ModelLoadError,
    ModelNotFoundError,
    ModelResourceError,
)
from src.core.models.base import BaseModel, ModelMetadata, ModelState
from src.services.model_registry import ModelRegistry


class MockModel(BaseModel):
    """Configurable mock model for registry testing."""

    def __init__(
        self,
        metadata: ModelMetadata,
        load_delay: float = 0.0,
        fail_load: bool = False,
    ) -> None:
        super().__init__(metadata)
        self.load_delay = load_delay
        self.fail_load = fail_load
        self.load_count = 0
        self.unload_count = 0

    def _load_weights(self, device: str) -> None:
        self.load_count += 1
        if self.load_delay > 0:
            time.sleep(self.load_delay)
        if self.fail_load:
            raise RuntimeError("Corrupted model checkpoint")

    def _unload_weights(self) -> None:
        self.unload_count += 1

    def _execute_inference(self, inputs: Any, **kwargs: Any) -> Any:
        return {"output": f"mock: {inputs}"}


@pytest.fixture
def mock_device_mgr() -> DeviceManager:
    return DeviceManager(preference="cpu")


@pytest.fixture
def registry(mock_device_mgr: DeviceManager) -> ModelRegistry:
    return ModelRegistry(device_mgr=mock_device_mgr, max_cached_models=2)


def create_meta(model_id: str, max_concurrency: int = 2) -> ModelMetadata:
    return ModelMetadata(
        model_id=model_id,
        name=f"Name {model_id}",
        task="sentiment",
        max_concurrency=max_concurrency,
    )


@pytest.mark.asyncio
async def test_registration_and_metadata_lookup(registry: ModelRegistry):
    """Verify model registration and metadata retrieval."""
    meta = create_meta("m-1")
    registry.register(meta, lambda: MockModel(meta))

    assert registry.is_registered("m-1")
    assert not registry.is_loaded("m-1")
    assert registry.get_metadata("m-1").name == "Name m-1"

    # Duplicate registration raises ValueError
    with pytest.raises(ValueError) as exc:
        registry.register(meta, lambda: MockModel(meta))
    assert "already registered" in str(exc.value)

    # Unknown model lookup raises ModelNotFoundError
    with pytest.raises(ModelNotFoundError):
        registry.get_metadata("unknown")


@pytest.mark.asyncio
async def test_lazy_loading_and_caching(registry: ModelRegistry):
    """Verify models are lazy loaded on demand and cached for subsequent requests."""
    meta = create_meta("m-1")
    instance = MockModel(meta)
    registry.register(meta, lambda: instance)

    assert registry.get_loaded_count() == 0
    # First request: lazy loads
    model1 = await registry.get_model("m-1")
    assert registry.get_loaded_count() == 1
    assert model1.is_ready()
    assert instance.load_count == 1

    # Second request: cache hit returns same instance without re-loading
    model2 = await registry.get_model("m-1")
    assert model1 is model2
    assert instance.load_count == 1


@pytest.mark.asyncio
async def test_concurrent_requests_same_model_no_duplicate_loads(registry: ModelRegistry):
    """Verify that concurrent requests for an unloaded model do not trigger duplicate loads."""
    meta = create_meta("slow-model")
    instance = MockModel(meta, load_delay=0.1)
    registry.register(meta, lambda: instance)

    # 10 simultaneous coroutines request the same model
    tasks = [registry.get_model("slow-model") for _ in range(10)]
    models = await asyncio.gather(*tasks)

    # All should receive the exact same ready instance
    for m in models:
        assert m is instance
        assert m.is_ready()

    # The load hook must have been called exactly ONCE
    assert instance.load_count == 1


@pytest.mark.asyncio
async def test_concurrent_loading_different_models(registry: ModelRegistry):
    """Verify loading different models happens concurrently without blocking each other."""
    m1_meta = create_meta("m1")
    m2_meta = create_meta("m2")
    inst1 = MockModel(m1_meta, load_delay=0.1)
    inst2 = MockModel(m2_meta, load_delay=0.1)

    registry.register(m1_meta, lambda: inst1)
    registry.register(m2_meta, lambda: inst2)

    start = time.perf_counter()
    res1, res2 = await asyncio.gather(
        registry.get_model("m1"),
        registry.get_model("m2"),
    )
    elapsed = time.perf_counter() - start

    assert res1.is_ready() and res2.is_ready()
    # Total elapsed should be around 0.1s - 0.2s, not serialized 0.2s+
    assert elapsed < 0.25


@pytest.mark.asyncio
async def test_lru_cache_eviction(registry: ModelRegistry):
    """Verify LRU cache evicts the least recently used model when max_cached_models is exceeded."""
    m1_meta = create_meta("m1")
    m2_meta = create_meta("m2")
    m3_meta = create_meta("m3")

    inst1 = MockModel(m1_meta)
    inst2 = MockModel(m2_meta)
    inst3 = MockModel(m3_meta)

    registry.register(m1_meta, lambda: inst1)
    registry.register(m2_meta, lambda: inst2)
    registry.register(m3_meta, lambda: inst3)

    # max_cached_models is 2
    # Load m1 and m2
    await registry.get_model("m1")
    await asyncio.sleep(0.01)
    await registry.get_model("m2")
    await asyncio.sleep(0.01)

    assert registry.is_loaded("m1")
    assert registry.is_loaded("m2")
    assert registry.get_loaded_count() == 2

    # Touch m1 with an inference to update its last_used_at
    await inst1.predict("test")
    await asyncio.sleep(0.01)

    # Now m2 is least recently used. Loading m3 must evict m2!
    await registry.get_model("m3")

    assert registry.get_loaded_count() == 2
    assert registry.is_loaded("m1")
    assert not registry.is_loaded("m2")
    assert registry.is_loaded("m3")
    assert inst2.unload_count == 1


@pytest.mark.asyncio
async def test_eviction_skips_busy_model(registry: ModelRegistry):
    """Verify that a model currently executing inference is protected from LRU eviction."""
    m1_meta = create_meta("m1")
    m2_meta = create_meta("m2")
    m3_meta = create_meta("m3")

    inst1 = MockModel(m1_meta)
    inst2 = MockModel(m2_meta)
    inst3 = MockModel(m3_meta)

    registry.register(m1_meta, lambda: inst1)
    registry.register(m2_meta, lambda: inst2)
    registry.register(m3_meta, lambda: inst3)

    # Load m1 and m2
    await registry.get_model("m1")
    await asyncio.sleep(0.01)
    await registry.get_model("m2")
    await asyncio.sleep(0.01)

    # Simulate an active long-running inference task on m1 (which is older than m2)
    # Manually acquire semaphore and increment active inferences on m1
    await inst1._semaphore.acquire()
    inst1._active_inferences = 1

    try:
        # Loading m3 needs to evict. Although m1 is older, m1 is busy, so m2 must be evicted!
        await registry.get_model("m3")

        assert registry.is_loaded("m1")
        assert not registry.is_loaded("m2")
        assert registry.is_loaded("m3")
        assert inst2.unload_count == 1
    finally:
        inst1._active_inferences = 0
        inst1._semaphore.release()


@pytest.mark.asyncio
async def test_all_models_busy_raises_resource_error(registry: ModelRegistry):
    """Verify that if all cached models are busy, loading another model raises ModelResourceError."""
    m1_meta = create_meta("m1")
    m2_meta = create_meta("m2")
    m3_meta = create_meta("m3")

    inst1 = MockModel(m1_meta)
    inst2 = MockModel(m2_meta)
    inst3 = MockModel(m3_meta)

    registry.register(m1_meta, lambda: inst1)
    registry.register(m2_meta, lambda: inst2)
    registry.register(m3_meta, lambda: inst3)

    await registry.get_model("m1")
    await registry.get_model("m2")

    # Mark BOTH m1 and m2 as busy
    inst1._active_inferences = 1
    inst2._active_inferences = 1

    try:
        with pytest.raises(ModelResourceError) as exc:
            await registry.get_model("m3")
        assert "all loaded models are currently executing active inference" in str(exc.value.message).lower()
    finally:
        inst1._active_inferences = 0
        inst2._active_inferences = 0


@pytest.mark.asyncio
async def test_loading_failure_handling_and_recovery(registry: ModelRegistry):
    """Verify that a failing model load sets FAILED state and can recover when fixed."""
    meta = create_meta("failing-model")
    fail_inst = MockModel(meta, fail_load=True)
    registry.register(meta, lambda: fail_inst)

    with pytest.raises(ModelLoadError):
        await registry.get_model("failing-model")

    assert not registry.is_loaded("failing-model")
    status = registry.get_model_status("failing-model")
    assert status["state"] == ModelState.FAILED.value
    assert "Corrupted" in str(status["error_message"])

    # Fix the failure on subsequent attempt
    fail_inst.fail_load = False
    recovered_model = await registry.get_model("failing-model")
    assert recovered_model.is_ready()
    assert registry.is_loaded("failing-model")


@pytest.mark.asyncio
async def test_unload_model_and_unload_all(registry: ModelRegistry):
    """Verify explicit model unloading and registry unload_all."""
    meta1 = create_meta("m1")
    meta2 = create_meta("m2")
    inst1 = MockModel(meta1)
    inst2 = MockModel(meta2)

    registry.register(meta1, lambda: inst1)
    registry.register(meta2, lambda: inst2)

    await registry.get_model("m1")
    await registry.get_model("m2")
    assert registry.get_loaded_count() == 2

    # Unload m1
    await registry.unload_model("m1")
    assert not registry.is_loaded("m1")
    assert registry.is_loaded("m2")
    assert registry.get_loaded_count() == 1

    # Unload all
    await registry.unload_all()
    assert registry.get_loaded_count() == 0
    assert inst2.unload_count == 1


@pytest.mark.asyncio
async def test_concurrent_loading_beyond_cache_limit_respects_max_cached():
    """Verify concurrent get_model calls cannot exceed max_cached_models while weights load simultaneously."""
    reg = ModelRegistry(device_mgr=DeviceManager(preference="cpu"), max_cached_models=2)
    m1_meta = create_meta("m1")
    m2_meta = create_meta("m2")
    m3_meta = create_meta("m3")

    reg.register(m1_meta, lambda: MockModel(m1_meta, load_delay=0.1))
    reg.register(m2_meta, lambda: MockModel(m2_meta, load_delay=0.1))
    reg.register(m3_meta, lambda: MockModel(m3_meta, load_delay=0.1))

    results = await asyncio.gather(
        reg.get_model("m1"),
        reg.get_model("m2"),
        reg.get_model("m3"),
        return_exceptions=True,
    )
    loaded = [r for r in results if not isinstance(r, Exception)]
    errors = [r for r in results if isinstance(r, Exception)]

    assert reg.get_loaded_count() <= 2
    assert len(loaded) == 2
    assert len(errors) == 1
    assert isinstance(errors[0], ModelResourceError)
