"""Centralized model registry and lifecycle manager.

Provides lazy loading, per-model initialization locks (preventing duplicate
concurrent downloads/initializations without serializing unrelated models),
LRU cache management with in-use eviction protection, explicit unloading,
and lifecycle health inspection.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from src.core.device import DeviceManager, device_manager
from src.core.exceptions import (
    ModelInUseError,
    ModelLoadError,
    ModelNotFoundError,
    ModelResourceError,
)
from src.core.logging import get_logger
from src.core.models.base import BaseModel, ModelMetadata, ModelState

logger = get_logger("nlpulse.services.model_registry")


@dataclass
class ModelRegistration:
    """Registration record pairing metadata with an instantiation factory."""

    metadata: ModelMetadata
    factory: Callable[[], BaseModel]
    registered_at: datetime
    last_error: Optional[str] = None


class ModelRegistry:
    """Thread-safe and async-safe registry managing model lifecycle and memory."""

    def __init__(
        self,
        device_mgr: Optional[DeviceManager] = None,
        max_cached_models: int = 3,
    ) -> None:
        """Initialize the model registry.

        Args:
            device_mgr: DeviceManager instance for hardware detection and resolution.
            max_cached_models: Maximum number of models held in memory simultaneously.
        """
        self.device_manager = device_mgr or device_manager
        self.max_cached_models = max(1, max_cached_models)

        self._registrations: Dict[str, ModelRegistration] = {}
        self._loaded_models: Dict[str, BaseModel] = {}
        self._loading_locks: Dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

        logger.info(
            "ModelRegistry initialized",
            extra={
                "max_cached_models": self.max_cached_models,
                "target_device": self.device_manager.target_device,
            },
        )

    def register(
        self,
        metadata: ModelMetadata,
        factory: Callable[[], BaseModel],
    ) -> None:
        """Register a model definition and its lazy instantiation factory.

        Args:
            metadata: Specifications and resource estimates for the model.
            factory: Callable that instantiates the concrete BaseModel subclass.

        Raises:
            ValueError: If a model with the same model_id is already registered.
        """
        model_id = metadata.model_id
        if model_id in self._registrations:
            raise ValueError(f"Model '{model_id}' is already registered.")

        self._registrations[model_id] = ModelRegistration(
            metadata=metadata,
            factory=factory,
            registered_at=datetime.now(timezone.utc),
        )
        self._loading_locks[model_id] = asyncio.Lock()
        logger.info(
            f"Model registered: {model_id} (task: {metadata.task}, max_concurrency: {metadata.max_concurrency})",
            extra={"model_id": model_id, "task": metadata.task},
        )

    def is_registered(self, model_id: str) -> bool:
        """Check if model is registered."""
        return model_id in self._registrations

    def is_loaded(self, model_id: str) -> bool:
        """Check if model is currently loaded in memory."""
        return model_id in self._loaded_models and self._loaded_models[model_id].is_ready()

    def get_metadata(self, model_id: str) -> ModelMetadata:
        """Retrieve model metadata by id.

        Raises:
            ModelNotFoundError: If model_id is not registered.
        """
        if model_id not in self._registrations:
            raise ModelNotFoundError(model_id)
        return self._registrations[model_id].metadata

    def _get_loading_lock(self, model_id: str) -> asyncio.Lock:
        """Get or initialize the loading lock for a specific model."""
        if model_id not in self._loading_locks:
            self._loading_locks[model_id] = asyncio.Lock()
        return self._loading_locks[model_id]

    async def get_model(
        self,
        model_id: str,
        preferred_device: Optional[str] = None,
    ) -> BaseModel:
        """Retrieve a model instance, lazily loading it on demand.

        Concurrent calls for the same unloaded model await the single loading
        operation without duplicate initialization. Concurrent calls for different
        models proceed in parallel without blocking each other.

        Args:
            model_id: Unique model identifier.
            preferred_device: Optional device override ('cuda', 'cpu', 'auto').

        Returns:
            Ready BaseModel instance.

        Raises:
            ModelNotFoundError: If model is not registered.
            ModelLoadError: If model fails to initialize or load weights.
            ModelResourceError: If all cached models are busy and cannot be evicted.
        """
        if model_id not in self._registrations:
            raise ModelNotFoundError(model_id)

        # Fast path: already loaded and ready
        if model_id in self._loaded_models:
            model = self._loaded_models[model_id]
            if model.is_ready():
                return model

        # Slow path: acquire model-specific loading lock to serialize initialization
        lock = self._get_loading_lock(model_id)
        async with lock:
            # Double-check if another concurrent request loaded it while waiting
            if model_id in self._loaded_models:
                model = self._loaded_models[model_id]
                if model.is_ready():
                    return model

            # Evict least recently used model if cache is full
            await self._ensure_cache_capacity(target_model_id=model_id)

            registration = self._registrations[model_id]
            device = self.device_manager.resolve_device(
                preferred_device or registration.metadata.target_device
            )

            logger.info(
                f"Lazy loading model '{model_id}' onto {device}",
                extra={"model_id": model_id, "device": device},
            )

            model = registration.factory()
            try:
                await model.load(device)
                self._loaded_models[model_id] = model
                registration.last_error = None
                return model
            except Exception as e:
                registration.last_error = str(e)
                if model_id in self._loaded_models:
                    del self._loaded_models[model_id]
                raise

    async def _ensure_cache_capacity(self, target_model_id: str) -> None:
        """Evict the least recently used idle model if loaded model count reaches limit.

        Args:
            target_model_id: Model ID about to be loaded.

        Raises:
            ModelResourceError: If cache is full and all loaded models are busy with inference.
        """
        async with self._registry_lock:
            if len(self._loaded_models) < self.max_cached_models:
                return

            # Find candidates for eviction (must not be the target, must not be busy)
            candidates: List[BaseModel] = [
                m for m_id, m in self._loaded_models.items()
                if m_id != target_model_id and not m.is_busy()
            ]

            if not candidates:
                logger.error(
                    "Model cache full and all loaded models are actively executing inference",
                    extra={
                        "loaded_count": len(self._loaded_models),
                        "max_cached": self.max_cached_models,
                        "target_model_id": target_model_id,
                    },
                )
                raise ModelResourceError(
                    f"Model cache limit ({self.max_cached_models}) reached and all loaded models "
                    "are currently executing active inference tasks. Cannot evict."
                )

            # Sort by last_used_at ascending (LRU)
            # If last_used_at is None, treat as oldest
            candidates.sort(
                key=lambda m: m.last_used_at or datetime.min.replace(tzinfo=timezone.utc)
            )
            lru_model = candidates[0]
            logger.info(
                f"Evicting LRU model '{lru_model.model_id}' from cache to free slot for '{target_model_id}'",
                extra={
                    "evicted_model": lru_model.model_id,
                    "target_model": target_model_id,
                    "last_used_at": lru_model.last_used_at.isoformat() if lru_model.last_used_at else None,
                },
            )

        # Unload outside of _registry_lock to avoid blocking other registry queries
        await self.unload_model(lru_model.model_id, force=False)

    async def unload_model(self, model_id: str, force: bool = False) -> None:
        """Unload a loaded model and release memory.

        Args:
            model_id: Identifier of model to unload.
            force: If True, unloads even if active inferences are running.

        Raises:
            ModelNotFoundError: If model_id is not registered.
            ModelInUseError: If active inferences are running and force is False.
        """
        if model_id not in self._registrations:
            raise ModelNotFoundError(model_id)

        lock = self._get_loading_lock(model_id)
        async with lock:
            if model_id not in self._loaded_models:
                logger.debug(f"Model '{model_id}' is not loaded; unload is a no-op")
                return

            model = self._loaded_models[model_id]
            if model.is_busy() and not force:
                raise ModelInUseError(model_id, model.active_inferences)

            await model.unload()
            async with self._registry_lock:
                self._loaded_models.pop(model_id, None)

            logger.info(
                f"Model '{model_id}' unloaded and removed from cache",
                extra={"model_id": model_id},
            )

    async def unload_all(self, force: bool = False) -> None:
        """Unload all currently loaded models (e.g. during application shutdown)."""
        logger.info(f"Unloading all {len(self._loaded_models)} loaded models from registry")
        model_ids = list(self._loaded_models.keys())
        for model_id in model_ids:
            try:
                await self.unload_model(model_id, force=force)
            except Exception as e:
                logger.warning(f"Error unloading model '{model_id}' during bulk unload: {e}")

    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Inspect the current status and metrics of a model."""
        if model_id not in self._registrations:
            raise ModelNotFoundError(model_id)

        reg = self._registrations[model_id]
        if model_id in self._loaded_models:
            status = self._loaded_models[model_id].to_dict()
        else:
            status = {
                "model_id": model_id,
                "name": reg.metadata.name,
                "task": reg.metadata.task,
                "version": reg.metadata.version,
                "state": ModelState.FAILED.value if reg.last_error else ModelState.UNLOADED.value,
                "device": None,
                "loaded_at": None,
                "last_used_at": None,
                "active_inferences": 0,
                "total_inferences": 0,
                "max_concurrency": reg.metadata.max_concurrency,
                "vram_estimate_mb": reg.metadata.vram_estimate_mb,
                "ram_estimate_mb": reg.metadata.ram_estimate_mb,
                "error_message": reg.last_error,
            }
        return status

    def list_models(self) -> List[Dict[str, Any]]:
        """Return status dictionaries for all registered models."""
        return [self.get_model_status(mid) for mid in sorted(self._registrations.keys())]

    def get_loaded_count(self) -> int:
        """Return number of currently loaded models."""
        return len(self._loaded_models)


# Module-level default singleton registry
model_registry = ModelRegistry()
