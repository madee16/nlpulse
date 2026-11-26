"""Standardized base model interface and lifecycle contracts.

Defines the core abstraction for all AI/NLP models in NLPulse, separating
model-specific inference execution from thread-safe lifecycle management,
concurrency control, device binding, and resource tracking.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel as PydanticBaseModel, Field

from src.core.device import device_manager
from src.core.exceptions import (
    ModelError,
    ModelInUseError,
    ModelInferenceError,
    ModelLoadError,
)
from src.core.logging import get_logger

logger = get_logger("nlpulse.core.models.base")


class ModelState(str, Enum):
    """Lifecycle states of an AI model instance."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    UNLOADING = "unloading"
    FAILED = "failed"


class ModelMetadata(PydanticBaseModel):
    """Metadata specification for model registration and resource planning."""

    model_id: str = Field(..., description="Unique alphanumeric identifier (e.g. sentiment-distilbert)")
    name: str = Field(..., description="Human-readable model name")
    task: str = Field(..., description="NLP task domain (e.g. sentiment, ner, zero-shot, summarization)")
    version: str = Field(default="1.0.0", description="Semantic model version")
    description: str = Field(default="", description="Detailed summary of model capability and backend")
    target_device: Optional[str] = Field(default=None, description="Preferred device override (cuda, cpu, auto)")
    vram_estimate_mb: float = Field(default=0.0, description="Estimated GPU VRAM footprint in megabytes")
    ram_estimate_mb: float = Field(default=250.0, description="Estimated system RAM footprint in megabytes")
    max_concurrency: int = Field(default=2, ge=1, description="Maximum concurrent inference requests allowed")
    timeout_seconds: float = Field(default=30.0, gt=0, description="Maximum timeout for inference execution")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary framework or architecture parameters")


class BaseModel(ABC):
    """Abstract base class establishing lifecycle and execution contracts for all NLP models."""

    def __init__(self, metadata: ModelMetadata) -> None:
        """Initialize base model with metadata and concurrency primitives.

        Args:
            metadata: Specifications and configuration for this model.
        """
        self.metadata = metadata
        self._state: ModelState = ModelState.UNLOADED
        self._device: Optional[str] = None
        self._loaded_at: Optional[datetime] = None
        self._last_used_at: Optional[datetime] = None
        self._error_message: Optional[str] = None

        # Concurrency and thread-safety
        self._active_inferences: int = 0
        self._total_inferences: int = 0
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(metadata.max_concurrency)
        self._state_lock: asyncio.Lock = asyncio.Lock()

    @property
    def model_id(self) -> str:
        """Unique model identifier."""
        return self.metadata.model_id

    @property
    def state(self) -> ModelState:
        """Current lifecycle state."""
        return self._state

    @property
    def device(self) -> Optional[str]:
        """Hardware device on which model is currently hosted."""
        return self._device

    @property
    def loaded_at(self) -> Optional[datetime]:
        """Timestamp when model completed loading."""
        return self._loaded_at

    @property
    def last_used_at(self) -> Optional[datetime]:
        """Timestamp of most recent inference invocation."""
        return self._last_used_at

    @property
    def active_inferences(self) -> int:
        """Number of inference requests currently executing."""
        return self._active_inferences

    @property
    def total_inferences(self) -> int:
        """Lifetime inference executions for this model instance."""
        return self._total_inferences

    @property
    def error_message(self) -> Optional[str]:
        """Last recorded error message, if any."""
        return self._error_message

    def is_ready(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._state == ModelState.READY

    def is_busy(self) -> bool:
        """Check if model has active inference tasks running."""
        return self._active_inferences > 0

    # -------------------------------------------------------------------------
    # Lifecycle Hooks (Abstract)
    # -------------------------------------------------------------------------

    @abstractmethod
    def _load_weights(self, device: str) -> None:
        """Synchronously load model weights, tokenizer, or pipelines onto device.

        Subclasses must implement this method. It is executed off-event-loop.

        Args:
            device: Target hardware device ('cuda' or 'cpu').
        """
        raise NotImplementedError

    @abstractmethod
    def _unload_weights(self) -> None:
        """Synchronously release model weights and tensors from device memory.

        Subclasses must implement this method to clean up references.
        """
        raise NotImplementedError

    @abstractmethod
    def _execute_inference(self, inputs: Any, **kwargs: Any) -> Any:
        """Synchronously perform forward pass or pipeline inference.

        Subclasses implement domain-specific NLP inference here.
        This method is invoked in a worker thread via asyncio.to_thread.

        Args:
            inputs: Raw text or structured input.
            **kwargs: Generation, threshold, or formatting parameters.

        Returns:
            Domain inference output.
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Concrete Lifecycle Methods
    # -------------------------------------------------------------------------

    async def load(self, device: str) -> None:
        """Initialize and load the model onto the specified device.

        Args:
            device: Hardware device to host the model ('cuda' or 'cpu').

        Raises:
            ModelLoadError: If weight loading or pipeline initialization fails.
        """
        async with self._state_lock:
            if self._state == ModelState.READY and self._device == device:
                logger.debug(f"Model '{self.model_id}' is already loaded on {device}")
                return

            self._state = ModelState.LOADING
            self._error_message = None
            logger.info(
                f"Loading model '{self.model_id}' onto {device}",
                extra={"model_id": self.model_id, "device": device},
            )

            try:
                # Offload heavy synchronous weight loading to worker thread
                await asyncio.to_thread(self._load_weights, device)
                self._device = device
                now = datetime.now(timezone.utc)
                self._loaded_at = now
                self._last_used_at = now
                self._state = ModelState.READY
                logger.info(
                    f"Model '{self.model_id}' loaded successfully on {device}",
                    extra={"model_id": self.model_id, "device": device},
                )
            except Exception as e:
                self._state = ModelState.FAILED
                self._error_message = str(e)
                self._device = None
                logger.error(
                    f"Failed to load model '{self.model_id}': {e}",
                    extra={"model_id": self.model_id, "error": str(e)},
                    exc_info=True,
                )
                raise ModelLoadError(self.model_id, str(e)) from e

    async def unload(self) -> None:
        """Unload the model and release host/device memory.

        Raises:
            ModelInUseError: If inference tasks are actively running.
        """
        async with self._state_lock:
            if self._state == ModelState.UNLOADED:
                return

            if self._active_inferences > 0:
                raise ModelInUseError(self.model_id, self._active_inferences)

            self._state = ModelState.UNLOADING
            logger.info(
                f"Unloading model '{self.model_id}'",
                extra={"model_id": self.model_id, "device": self._device},
            )

            try:
                await asyncio.to_thread(self._unload_weights)
            except Exception as e:
                logger.warning(f"Error during weight unload for '{self.model_id}': {e}")
            finally:
                if self._device == "cuda":
                    device_manager.empty_cuda_cache()

                self._device = None
                self._loaded_at = None
                self._state = ModelState.UNLOADED
                logger.info(
                    f"Model '{self.model_id}' successfully unloaded",
                    extra={"model_id": self.model_id},
                )

    def _execute_batch_inference(self, inputs: Any, **kwargs: Any) -> Any:
        """Synchronously perform batched inference across multiple items.

        Default implementation executes `_execute_inference` sequentially inside a single
        worker slot. Subclasses whose underlying framework supports vectorized tensor
        batching should override this method.
        """
        if not isinstance(inputs, list):
            raise ValueError(f"Expected list input for batch inference, got {type(inputs).__name__}")
        return [self._execute_inference(item, **kwargs) for item in inputs]

    async def _run_worker(self, fn: Any, inputs: Any, **kwargs: Any) -> Any:
        """Execute a synchronous model callable under M3 semaphore and lifecycle guards."""
        if self._state != ModelState.READY:
            raise ModelError(
                f"Cannot execute inference on model '{self.model_id}': state is {self._state.value}",
                status_code=503,
            )

        # Acquire concurrency slot (bounded by max_concurrency)
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.metadata.timeout_seconds,
            )
        except asyncio.TimeoutError as e:
            raise ModelInferenceError(
                self.model_id,
                f"Inference concurrency queue timed out after {self.metadata.timeout_seconds}s",
            ) from e

        self._active_inferences += 1
        self._last_used_at = datetime.now(timezone.utc)

        # Worker thread lifecycle management:
        # Cancelling an await on asyncio.to_thread does NOT terminate the OS thread.
        # We ensure _active_inferences and _semaphore are released ONLY when the worker thread
        # actually completes, preventing model unload or concurrency overflow during timeout.
        worker_task = asyncio.ensure_future(
            asyncio.to_thread(fn, inputs, **kwargs)
        )

        slot_released = False
        worker_timed_out = False

        def _cleanup_worker(fut: asyncio.Future) -> None:
            nonlocal slot_released
            if not slot_released:
                slot_released = True
                self._active_inferences = max(0, self._active_inferences - 1)
                self._semaphore.release()
                # If caller already timed out, handle any deferred OOM cleanup here
                if worker_timed_out and not fut.cancelled():
                    try:
                        exc = fut.exception()
                        if exc and ("out of memory" in str(exc).lower() or "cuda oom" in str(exc).lower()):
                            device_manager.handle_cuda_oom()
                    except (asyncio.CancelledError, Exception):
                        pass

        worker_task.add_done_callback(_cleanup_worker)

        try:
            # Shield worker_task so timeout cancellation does not cancel the task future
            result = await asyncio.wait_for(
                asyncio.shield(worker_task),
                timeout=self.metadata.timeout_seconds,
            )
            self._total_inferences += 1
            return result
        except asyncio.TimeoutError as e:
            worker_timed_out = True
            logger.error(
                f"Inference execution for '{self.model_id}' exceeded {self.metadata.timeout_seconds}s timeout "
                f"(underlying thread remains accounted for in active_inferences until it terminates)"
            )
            raise ModelInferenceError(
                self.model_id,
                f"Inference execution timed out after {self.metadata.timeout_seconds}s",
            ) from e
        except Exception as e:
            err_msg = str(e)
            # Detect CUDA out-of-memory specifically
            if "out of memory" in err_msg.lower() or "cuda oom" in err_msg.lower():
                device_manager.handle_cuda_oom()
                logger.critical(f"CUDA Out-of-Memory during inference on '{self.model_id}'")
            logger.error(
                f"Inference failed on '{self.model_id}': {e}",
                extra={"model_id": self.model_id, "error": err_msg},
                exc_info=True,
            )
            raise ModelInferenceError(self.model_id, err_msg) from e

    async def predict(self, inputs: Any, **kwargs: Any) -> Any:
        """Execute inference with concurrency guarding and async event-loop protection.

        Args:
            inputs: Text or structured payload for inference.
            **kwargs: Extra parameters passed to the model inference method.

        Returns:
            Output from `_execute_inference`.

        Raises:
            ModelError: If the model is not in the READY state.
            ModelInferenceError: If inference execution fails or times out.
        """
        return await self._run_worker(self._execute_inference, inputs, **kwargs)

    async def predict_batch(self, inputs: Any, **kwargs: Any) -> Any:
        """Execute batched inference preserving M3 concurrency, timeout, and OOM guards."""
        if self._state != ModelState.READY:
            raise ModelError(
                f"Cannot execute inference on model '{self.model_id}': state is {self._state.value}",
                status_code=503,
            )
        if isinstance(inputs, list) and not inputs:
            return []
        return await self._run_worker(self._execute_batch_inference, inputs, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Produce a comprehensive monitoring status dictionary."""
        return {
            "model_id": self.model_id,
            "name": self.metadata.name,
            "task": self.metadata.task,
            "version": self.metadata.version,
            "state": self._state.value,
            "device": self._device,
            "loaded_at": self._loaded_at.isoformat() if self._loaded_at else None,
            "last_used_at": self._last_used_at.isoformat() if self._last_used_at else None,
            "active_inferences": self._active_inferences,
            "total_inferences": self._total_inferences,
            "max_concurrency": self.metadata.max_concurrency,
            "vram_estimate_mb": self.metadata.vram_estimate_mb,
            "ram_estimate_mb": self.metadata.ram_estimate_mb,
            "error_message": self._error_message,
        }
