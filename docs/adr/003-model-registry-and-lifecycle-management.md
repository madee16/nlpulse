# ADR-003: Model Registry and Lifecycle Management

- **Status:** Accepted (Current)

## Context

NLPulse wraps five distinct Hugging Face Transformer / Sentence-Transformer pipelines (DistilBERT Sentiment, BERT NER, DistilBART Zero-Shot, MiniLM Embeddings, DistilBART CNN Summarization). Loading all five eagerly consumes significant system RAM and can exceed 6 GB laptop VRAM budgets, while loading models per HTTP request creates multi-second latency spikes and concurrency races.

## Decision

Manage all local Hugging Face models through a centralized **`ModelRegistry`** (`src/services/model_registry.py`) and **`BaseModel`** contract (`src/core/models/base.py`):
- **Lazy Registration on Startup:** FastAPI `lifespan()` registers `ModelMetadata` and instantiation factories without downloading or loading weights (`MODEL_ENABLE_WARMUP` remains a reserved config field, not implemented).
- **Per-Model Load Locking + Slot Reservation:** `_loading_locks[model_id]` deduplicates concurrent loads of the same model while letting distinct models load concurrently, bounded under `_registry_lock` by `_pending_loads` and `_evicting_models` so `len(_loaded_models) + len(_pending_loads) <= max_cached_models` (default `3`).
- **LRU Eviction with In-Use Protection:** Evicts the least recently used idle model (`not is_busy()`); returns HTTP `503` (`ModelResourceError`) if all cached slots are actively running inference.
- **Shielded Off-Loop Worker Execution:** `BaseModel._run_worker` enforces per-model `asyncio.Semaphore(max_concurrency)`, executes synchronous inference via `asyncio.to_thread`, and shields the `asyncio.Future` so timeouts release `_active_inferences` and the semaphore only in the `add_done_callback` handler once the OS thread finishes.

## Alternatives Considered

1. **Eager startup loading of all 5 models:** Simplifies request hot paths, but crashes or thrashes swap on memory-constrained developer machines (`6 GB VRAM / 16 GB RAM`) and breaks fast offline/test startup.
2. **Unbounded global dictionary of loaded pipelines:** Simple caching, but concurrent distinct model loads can exceed memory budgets and unsafe unloads could corrupt mid-flight inference threads.

## Consequences

- **Positive:** Fast application boot with zero startup weight downloads, strict upper bound on resident model count, safe concurrent requests without duplicate model initialization, and clean shutdown via `unload_all()`.
- **Negative:** First request targeting an unloaded model absorbs weight download/load latency; when all `max_cached_models` slots are busy simultaneously, incoming requests for an uncached 4th model fail fast with `503`.

## Current Status

Implemented and verified via `tests/unit/test_model_registry.py`, `tests/unit/test_model_abstraction.py`, `tests/unit/test_m3_lifecycle_hardening.py`, and `tests/integration/test_models_api.py` (with `MODEL_MAX_CACHED` concurrency race fix included).
