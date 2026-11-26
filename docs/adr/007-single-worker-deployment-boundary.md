# ADR-007: Single-Worker Deployment Boundary

- **Status:** Accepted (Current Architectural Boundary)

## Context

Both `SlidingWindowRateLimiter` (`src/core/rate_limiter.py`) and `QuotaReservationManager` (`src/services/usage_service.py`), along with `ModelRegistry` (`src/services/model_registry.py`), store state in process-local Python memory guarded by striped `asyncio.Lock` primitives. When evaluating multi-worker Uvicorn (`--workers > 1`) during readiness verification, spawning N independent OS worker processes causes N uncoordinated copies of rate-limit counters, N independent reservation trackers, and N duplicate copies of heavy PyTorch model caches competing for the same host RAM / 6 GB VRAM.

## Decision

Explicitly define and document **single-worker ASGI execution** (`uvicorn src.main:app --host 0.0.0.0 --port 8000`, one process) as the supported deployment model for NLPulse, without adding Redis, Celery, or distributed coordination infrastructure to the repository.

## Alternatives Considered

1. **Add Redis + Lua scripts for shared rate limiting and distributed quota leases:** Solves cross-worker coordination, but adds another external stateful service, connection failure modes, and Redis dependency overhead unsupported by the current lean local-first single-node footprint.
2. **Run multi-worker Uvicorn anyway with process-local limiters:** Causes effective rate limits to multiply by worker count (`N * 60 req/min`), breaks in-flight quota reservation consistency across workers, and multiplies Transformer memory consumption by `N` (`3 * N` cached models).

## Consequences

- **Positive:** Zero extra infrastructure dependencies (only PostgreSQL 16 required; Ollama optional), maximal PyTorch model weight sharing inside one asyncio event loop + worker threadpool, and predictable single-node memory footprint bounded by `MODEL_MAX_CACHED`.
- **Negative:** Vertical scaling on CPU-bound PyTorch tasks within one process is constrained by `MODEL_DEFAULT_CONCURRENCY` (`2`) and GIL contention across native threads (though PyTorch/Transformers release the GIL during C++/LibTorch tensor kernels); horizontal multi-worker scaling requires externalizing rate limiting and quota reservation (e.g., Redis) plus routing inference or model serving out-of-process.

## Current Status

Verified for single-worker concurrency (`tests/integration/test_postgres_real.py`, `tests/integration/test_batch_api.py`, `tests/integration/test_m8_security_and_reliability.py`) and documented explicitly as the supported deployment boundary.
