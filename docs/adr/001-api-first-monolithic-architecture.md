# ADR-001: API-First Monolithic Architecture

- **Status:** Accepted (Current)

## Context

NLPulse provides developer-facing REST access to six NLP capabilities (plus batch variants), tenant/API-key management, usage metering, and model lifecycle inspection. Early design had to choose how to structure compute-bound PyTorch pipelines, database persistence, and optional generative inference for a maintainable single-developer AI Engineering platform targeting laptop/single-node hardware (16 GB RAM / 6 GB VRAM class).

## Decision

Build NLPulse as a modular, API-first FastAPI monolith (`src/main.py`) organized into clear internal layers:
- **Transport & Security (`src/api/`):** Routers, header auth, scope checks, rate limits, quota reservation dependencies.
- **Business & Orchestration (`src/services/`):** Domain services (`NLPService`, `BatchService`, `ExtractionService`, `ModelRegistry`, `UsageService`, `APIKeyService`, `DeveloperService`, `OllamaClient`).
- **Model Execution (`src/core/models/`):** Hardware-aware lazy model wrappers offloaded to worker threads via `asyncio.to_thread`.
- **Persistence (`src/models/`, `src/db/`):** Async SQLAlchemy ORM backed by PostgreSQL 16 and Alembic.

## Alternatives Considered

1. **Microservices per NLP capability:** Independent FastAPI containers for Sentiment, NER, Summarization, and Auth behind an API gateway. Rejected due to disproportionate operational overhead, duplicated weight caching per container, and unnecessary network serialization latency.
2. **Distributed task queue workers (Celery/RQ + broker) for all requests:** Synchronous endpoints returning job IDs plus polling endpoints. Rejected for the core REST API because short transformer/batch payloads complete cleanly within single HTTP request bounds using bounded async worker pools.

## Consequences

- **Positive:** Single deployable process with unified OpenAPI documentation (`/docs`), in-process sharing of loaded Transformer weights across requests and batch endpoints, zero RPC serialization overhead between auth/metering and inference, and fast offline integration tests.
- **Negative:** All in-process Transformer models share one process memory space (`MODEL_MAX_CACHED=3` LRU cache mitigates RAM/VRAM pressure), and horizontal multi-process scaling requires externalizing process-local coordination state (see [ADR-007](./007-single-worker-deployment-boundary.md)).

## Current Status

Implemented and active across `src/`. All 26 OpenAPI routes execute inside the unified FastAPI application.
