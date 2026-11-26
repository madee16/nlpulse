# NLPulse — Portfolio Overview

**NLPulse** is an asynchronous, developer-facing **NLP-as-a-Service REST platform** combining Hugging Face Transformer inference, optional local LLM structured extraction (Ollama), API key security, rate limiting, atomic quota enforcement, and PostgreSQL persistence in a single resource-bounded Python 3.12 backend.

---

## Capabilities at a Glance

- **Core & Advanced Transformer Pipelines:**
  - **Sentiment Analysis** (`distilbert/distilbert-base-uncased-finetuned-sst-2-english`) with full class score distributions.
  - **Named Entity Recognition** (`dslim/bert-base-NER`) with subword aggregation and character span offset alignment.
  - **Zero-Shot Text Classification** (`valhalla/distilbart-mnli-12-1`) supporting softmax single-label and independent multi-label scoring.
  - **Semantic Keyword & Keyphrase Extraction** (`sentence-transformers/all-MiniLM-L6-v2`) using L2-normalized mean-pooled embeddings and deterministic cosine-similarity ranking.
  - **Abstractive Summarization** (`sshleifer/distilbart-cnn-12-6`) with token-bounded sentence chunking and hierarchical multi-round consolidation.
- **Local Generative Extraction (Optional Ollama):** Grounded JSON extraction supporting custom JSON Schemas or field lists (`qwen2.5:3b` default), recursive schema validation, context-aware placeholder normalization (`null` vs. literal source occurrences), and token-weighted cost metering.
- **Resource-Conscious Batch Processing:** All six capabilities support batch endpoints (`/api/v1/nlp/batch/*`) featuring micro-batched tensor forward passes, cross-document deduplicated embedding computation, bounded coroutine pools, per-item validation isolation, stable ordering, and success-only quota settlement.

---

## Architecture & Technical Stack

- **API & Validation:** Python 3.12, FastAPI, Pydantic v2, Pydantic Settings, `jsonschema` (Draft 2020-12), Uvicorn (single-worker ASGI).
- **Local Playground:** React 19, Vite, TypeScript (`frontend/`, standalone optional developer UI).
- **Persistence:** PostgreSQL 16, SQLAlchemy 2.x async (`asyncpg`), Alembic async migrations (`pool_pre_ping`, check constraints, cascade/audit foreign keys).
- **ML & LLM Runtime:** PyTorch, Hugging Face Transformers, `httpx.AsyncClient` for decoupled optional Ollama `/api/generate` calls.
- **Testing & Packaging:** Pytest, `pytest-asyncio`, `HTTPX` (`ASGITransport`), `aiosqlite`, Vitest, Docker Compose (PostgreSQL 16 container orchestration).

---

## Key Engineering Highlights

1. **Lifecycle-Safe Model Cache (`ModelRegistry` + `BaseModel`):** Lazy loading with per-model initialization locks, reserved-slot accounting across concurrent distinct loads (`MODEL_MAX_CACHED=3` LRU eviction), `asyncio.Semaphore` concurrency limits, and shielded `asyncio.to_thread` worker cleanup so timeouts never trigger unsafe mid-flight model unloading.
2. **Deadlock-Free Atomic Quota Enforcement:** Audited concurrent PostgreSQL 16 behavior, eliminated an AB-BA deadlock between `SELECT ... FOR UPDATE` row locks and in-process locks, closed single-request TOCTOU races (`reserve_single_quota`), and committed usage records synchronously alongside reservation release inside the same per-tenant striped lock.
3. **Security, Admission Control & Extraction Validation:** SHA-256 hashed API keys with single-display creation, non-delegable granular scopes, per-IP registration rate limiting with explicit `TRUSTED_PROXY_IPS` handling, tenant-isolated management endpoints, full Draft 2020-12 JSON Schema validation with raw LLM output redaction in logs/errors, and sanitized `503`/`500` error handling.

---

## Verification Evidence & Scope

- **Offline Unit & Integration Suite (`pytest -m "not smoke"`):** Comprehensive unit, HTTP integration, security regression, extraction schema/redaction, migration env-safety, and lifecycle failure tests verified cleanly using SQLite (`aiosqlite`) and mocked model/Ollama pipelines (while `@pytest.mark.postgres` tests skip cleanly unless `NLPULSE_TEST_POSTGRES_URL` is exported and `@pytest.mark.smoke` live-HF tests are deselected by default).
- **Real PostgreSQL 16 Verification (`@pytest.mark.postgres`):** Automated opt-in verification against native **PostgreSQL 16.15** (via WSL) covers Alembic `upgrade head`, `alembic check` (`No new upgrade operations detected.`), reversible `downgrade base` cleanup guarded by test-DB validation, SQL `CHECK` constraints, rollbacks, `CASCADE`/`SET NULL` foreign keys, and mixed concurrent batch + single-request quota locking with zero deadlocks.
- **Local Frontend Verification:** TypeScript checks, Vitest suites, Vite production build, and live Vite preview + Uvicorn CORS/health/registration/models/error probes executed without mocking backend contracts.

---

## Honest Limitations

- **Single-Worker Process Boundary:** API rate limiting, public registration admission rate limiting (`registration_rate_limiter`), and in-flight quota reservation live in process memory, so single-worker Uvicorn is the supported runtime model without an external shared coordinator (not distributed abuse protection).
- **Hardware & Live Model Verification:** PyTorch in `.venv` is CPU-only (`cuda_available=False`), so physical CUDA kernel execution is unverified despite CUDA detection/OOM telemetry logic being unit-tested. Live Hugging Face downloads (`-m smoke`) and end-to-end live Ollama extraction generations remain unverified in automated tests (live server probing hit a local Ollama daemon returning `404` when `qwen2.5:3b` was not pulled).
- **Containerized DB Check:** `docker compose config` passes validation, while local PostgreSQL 16 verification ran natively via WSL rather than through Docker Desktop.
