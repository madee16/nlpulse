# NLPulse — Intelligent Text Analysis API (NLP-as-a-Service)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-e92063.svg)](https://docs.pydantic.dev/)
[![PostgreSQL 16](https://img.shields.io/badge/PostgreSQL-16-336791.svg?logo=postgresql)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NLPulse** is a high-performance developer-facing REST API providing Natural Language Processing as a Service. It delivers end-to-end NLP pipelines including sentiment analysis, named entity recognition (NER), zero-shot text classification, keyword extraction, text summarization, and LLM-powered structured extraction.

---

## Current Status: M1–M8 Complete ✅

- ✅ **M1: Foundations & Database** — Project structure, PostgreSQL 16, async SQLAlchemy 2.x session lifecycle, models (`Developer`, `APIKey`, `UsageRecord`), and Alembic async migrations.
- ✅ **M2: Security & Authentication** — API keys, SHA-256 constant-time hashing, scope enforcement, sliding window rate limiting, monthly database quotas, and tenant isolation.
- ✅ **M3: Model Lifecycle Management** — Lazy model initialization, thread-safe `ModelRegistry`, LRU eviction with in-use protection, hardware detection (`DeviceManager` for CUDA GPU/CPU), per-model concurrency control, event-loop offloading via `asyncio.to_thread` with shielded thread lifecycle retention, and FastAPI lifespan integration.
- ✅ **M4: Core NLP** — Sentiment analysis (`distilbert-base-uncased-finetuned-sst-2-english`), Named Entity Recognition (`bert-base-NER` with subword BPE token aggregation), and Zero-Shot text classification (`distilbart-mnli-12-1` with single-label softmax & multi-label sigmoid scoring). Complete end-to-end API integration with rate limiting, monthly quota enforcement, and per-tenant usage metering.
- ✅ **M5: Advanced NLP** — Dense embedding keyword and keyphrase extraction (`sentence-transformers/all-MiniLM-L6-v2` with cosine similarity ranking and customizable n-gram / stopword filtering) and abstractive document summarization (`sshleifer/distilbart-cnn-12-6` with token-bounded sentence chunking and hierarchical consolidation). Fully integrated with LRU cache eviction, scopes (`keywords:read`, `summarization:read`), rate limits, quotas, and atomic usage metering.
- ✅ **M6: Generative NLP** — Local LLM integration via asynchronous `OllamaClient`, structured information extraction (`POST /api/v1/nlp/extraction`) adhering to JSON Schema or field specifications, programmatic schema validation with null fallbacks for unstated fields, strict factual grounding, configurable instruct models (e.g. `qwen2.5:3b`, `llama3.2:3b`), and transparent token-based cost unit metering.
- ✅ **M7: Batch Processing** — Multi-item REST endpoints across all 6 NLP capabilities (`/api/v1/nlp/batch/*`) featuring micro-batched tensor inference, cross-document deduplicated embedding batching, bounded concurrent worker pools, partial-failure isolation with stable ordering, atomic quota reservation, and accurate per-item + aggregate metering.
- ✅ **M8: Integration & Verification** — Cross-milestone integration audit, security hardening (strict empty-scope rejection, `/keys` & `/usage` scopes, atomic single-request quota reservation, safe `503`/`500` handling, zero-shot input bounds, `:path` routing for slashed HF model IDs), PostgreSQL 16.15 AB-BA deadlock elimination, and regression/reliability verification.
- ⏳ **M9: Documentation & Portfolio** — Next: Portfolio architecture documentation, ADRs, and final presentation.

---

## Tech Stack

- **Language:** Python 3.12
- **Framework:** FastAPI
- **Configuration & Validation:** Pydantic v2, Pydantic Settings, email-validator
- **Database:** PostgreSQL 16 (orchestrated via Docker Compose)
- **Database Driver / ORM:** SQLAlchemy 2.x (async), Asyncpg
- **Migrations:** Alembic (async)
- **Testing:** Pytest, pytest-asyncio, HTTPX, aiosqlite
- **Server:** Uvicorn (ASGI)

---

## Project Structure

```text
02-NLPulse/
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore for Python, venv, caches, weights
├── Dockerfile                # Container build specification
├── alembic.ini               # Alembic configuration
├── docker-compose.yml        # PostgreSQL 16 local orchestration
├── pyproject.toml            # Project metadata and build configuration
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── migrations/               # Alembic migration revisions
│   ├── env.py                # Async migration runner (zero hardcoded secrets)
│   ├── script.py.mako
│   └── versions/             # Migration scripts
├── src/
│   ├── __init__.py           # Application package root
│   ├── main.py               # FastAPI entry point, lifespan & security headers
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py           # Dependencies (get_db, auth, scopes, quota, rate-limit)
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── batch.py      # M7 batch NLP endpoints (/api/v1/nlp/batch/*)
│   │       ├── developers.py # Developer registration & profile endpoints
│   │       ├── health.py     # Health check endpoint (/api/v1/health)
│   │       ├── keys.py       # API key issuance, listing, and revocation
│   │       ├── models.py     # Model lifecycle & hardware telemetry routes
│   │       ├── nlp.py        # M4-M6 NLP inference endpoints
│   │       ├── protected.py  # Scope & quota protected verification routes
│   │       └── usage.py      # Monthly quota summary endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py         # Pydantic Settings & environment variables
│   │   ├── nlp_utils.py      # Stopword filtering, n-gram extraction, and chunking
│   │   ├── device.py         # Hardware detection, CUDA GPU/CPU & VRAM telemetry
│   │   ├── exceptions.py     # Domain exceptions (Auth, Quota, RateLimit, Model, Device)
│   │   ├── logging.py        # Structured JSON & console logging
│   │   ├── models/           # Standardized model lifecycle base abstractions
│   │   │   ├── __init__.py
│   │   │   ├── base.py       # BaseModel, ModelMetadata, ModelState, concurrency guards
│   │   │   └── nlp.py        # Sentiment, NER, and Zero-Shot model implementations
│   │   ├── rate_limiter.py   # Thread-safe striped sliding window rate limiter
│   │   └── security.py       # SHA-256 API key hashing & constant-time comparison
│   ├── db/
│   │   ├── __init__.py
│   │   ├── base.py           # Declarative base & metadata registry
│   │   └── session.py        # Async engine & sessionmaker
│   ├── models/
│   │   ├── __init__.py       # Database model exports
│   │   ├── api_key.py        # APIKey model & hash indexing
│   │   ├── developer.py      # Developer tenant root model
│   │   └── usage_record.py   # UsageRecord metering & audit model
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── api_key.py        # API key request/response schemas
│   │   ├── batch.py          # Batch request/response & partial-failure schemas (M7)
│   │   ├── developer.py      # Developer registration/profile schemas
│   │   ├── model.py          # Model telemetry & hardware metrics schemas
│   │   ├── nlp.py            # NLP requests, responses, and entity schemas
│   │   ├── extraction.py     # Schema definitions & extraction schemas (M6)
│   │   └── usage.py          # Usage summary schemas
│   └── services/
│       ├── __init__.py
│       ├── api_key_service.py   # Key issuance, hashing, and auth
│       ├── batch_service.py     # M7 batch orchestration, concurrency & metering
│       ├── developer_service.py # Tenant management & quota assignment
│       ├── extraction_service.py# Generative structured extraction service (M6)
│       ├── model_registry.py    # Lazy loading, LRU caching, eviction & locks
│       ├── nlp_service.py       # Core NLP orchestration & atomic usage metering
│       ├── ollama_client.py     # Asynchronous local Ollama HTTP client (M6)
│       └── usage_service.py     # Metering, quota reservation & monthly checks
└── tests/
    ├── __init__.py
    ├── conftest.py           # Pytest fixtures (async client & in-memory test DB)
    ├── integration/
    │   ├── __init__.py
    │   ├── test_auth_api.py   # End-to-end API registration, auth, and scopes
    │   ├── test_batch_api.py  # M7 batch API integration & quota race tests
    │   ├── test_extraction_api.py # M6 structured extraction API tests
    │   ├── test_m5_nlp_api.py # Keywords and summarization API tests
    │   ├── test_m8_security_and_reliability.py # M8 security, validation & reliability tests
    │   ├── test_migrations.py # Alembic upgrade/downgrade integration tests
    │   ├── test_models_api.py # Model registry and hardware telemetry API tests
    │   ├── test_nlp_api.py    # Core NLP (sentiment, ner, classification) API tests
    │   └── test_postgres_real.py # Opt-in PostgreSQL 16 concurrency & constraint tests
    ├── smoke/
    │   ├── __init__.py
    │   └── test_real_models_smoke.py # Opt-in live Hugging Face smoke tests
    └── unit/
        ├── __init__.py
        ├── test_batch_models.py  # M7 model-level batch inference tests
        ├── test_batch_quota.py   # M7 atomic quota reservation & settlement tests
        ├── test_batch_schemas.py # M7 batch schema envelope & validation tests
        ├── test_batch_service.py # M7 BatchService orchestration & cancellation tests
        ├── test_config.py     # Configuration tests
        ├── test_device.py     # Hardware detection & CUDA fallback tests
        ├── test_extraction_service.py # M6 extraction schema validation tests
        ├── test_health.py     # Health check endpoint tests
        ├── test_logging.py    # Structured logging tests
        ├── test_m3_lifecycle_hardening.py # Thread retention & CUDA OOM tests
        ├── test_model_abstraction.py # BaseModel lifecycle, semaphore & offloading tests
        ├── test_model_registry.py    # Lazy load, per-model lock & LRU cache tests
        ├── test_models.py     # Model relationships, constraints, cascades
        ├── test_nlp_models.py # Sentiment, NER, and Zero-Shot model unit tests
        ├── test_nlp_service.py# NLPService inference & metering unit tests
        ├── test_nlp_utils.py  # N-gram candidate extraction, stopwords, chunking tests
        ├── test_m5_models.py  # EmbeddingModel and SummarizationModel unit tests
        ├── test_m5_nlp_service.py # Keyword extraction & summarization service tests
        ├── test_m5_lifecycle.py   # M5 lazy loading, concurrency & LRU eviction tests
        ├── test_ollama_client.py  # M6 Ollama HTTP client unit tests
        ├── test_rate_limiter.py # Sliding window rate limiter tests
        ├── test_security.py   # Cryptographic API key tests
        ├── test_security_audit_regression.py # Security audit regression tests
        ├── test_services.py   # Business services unit tests
        └── test_tenant_isolation.py # Multi-tenant isolation tests
```

---

## API Endpoints (Milestones 1–8)

| Method | Endpoint | Auth Required | Scopes | Description |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/` | No | - | API root & discovery metadata |
| `GET` | `/api/v1/health` | No | - | Service health check |
| `POST` | `/api/v1/developers` | No | - | Register developer (enforces free tier & returns bootstrap API key) |
| `GET` | `/api/v1/developers/me` | `X-API-Key` | - | View authenticated developer profile & quota |
| `POST` | `/api/v1/keys` | `X-API-Key` | `keys:write` / `admin` | Create a child API key (non-empty scopes subset of caller's key) |
| `GET` | `/api/v1/keys` | `X-API-Key` | `keys:read` / `keys:write` / `admin` | List all API keys for tenant |
| `DELETE` | `/api/v1/keys/{key_id}` | `X-API-Key` | `keys:write` / `admin` | Revoke an API key |
| `GET` | `/api/v1/usage/summary` | `X-API-Key` | `usage:read` / `admin` | Monthly quota consumption & remaining requests |
| `POST` | `/api/v1/nlp/sentiment` | `X-API-Key` | `sentiment:read` | DistilBERT sentiment analysis & confidence distribution |
| `POST` | `/api/v1/nlp/ner` | `X-API-Key` | `ner:read` | BERT named entity recognition with character span offsets |
| `POST` | `/api/v1/nlp/classification` | `X-API-Key` | `classification:read` | DistilBART zero-shot classification (single or multi-label) |
| `POST` | `/api/v1/nlp/keywords` | `X-API-Key` | `keywords:read` | MiniLM dense embedding keyword & keyphrase ranking |
| `POST` | `/api/v1/nlp/summarization` | `X-API-Key` | `summarization:read` | DistilBART abstractive text summarization with chunking |
| `POST` | `/api/v1/nlp/extraction` | `X-API-Key` | `extraction:read` | Local LLM grounded structured information extraction (M6) |
| `POST` | `/api/v1/nlp/batch/sentiment` | `X-API-Key` | `sentiment:read` | Micro-batched sentiment analysis with partial-failure isolation (M7) |
| `POST` | `/api/v1/nlp/batch/ner` | `X-API-Key` | `ner:read` | Micro-batched NER across multiple texts (M7) |
| `POST` | `/api/v1/nlp/batch/classification` | `X-API-Key` | `classification:read` | Grouped batched zero-shot classification (M7) |
| `POST` | `/api/v1/nlp/batch/keywords` | `X-API-Key` | `keywords:read` | Cross-document deduplicated embedding keyword batching (M7) |
| `POST` | `/api/v1/nlp/batch/summarization` | `X-API-Key` | `summarization:read` | Bounded concurrent hierarchical summarization batch (M7) |
| `POST` | `/api/v1/nlp/batch/extraction` | `X-API-Key` | `extraction:read` | Bounded concurrent Ollama structured extraction batch (M7) |
| `GET` | `/api/v1/models` | `X-API-Key` | `models:read` / `admin` | List registered models and operational status |
| `GET` | `/api/v1/models/{model_id:path}` | `X-API-Key` | `models:read` / `admin` | Retrieve model telemetry and loading state (supports HF `org/model` IDs) |
| `POST` | `/api/v1/models/{model_id:path}/unload` | `X-API-Key` | `admin` | Explicitly unload model and release GPU/RAM |
| `GET` | `/api/v1/hardware` | `X-API-Key` | `admin` | Inspect compute hardware, GPU availability, and VRAM |

> **Scope Compatibility:** All `/api/v1/nlp/*` and `/api/v1/nlp/batch/*` endpoints also accept `nlp:read`, `analyze:read` (legacy compatibility), and `admin`. Extraction endpoints also accept `generate:read`.

---

## Advanced NLP Architecture (Milestone 5)

### 1. Keyword & Keyphrase Extraction Methodology
NLPulse implements an embedding-based semantic keyword extractor inspired by KeyBERT:
1. **Candidate N-Gram Generation:** Extracts words and phrases bounded by configurable n-gram ranges ($1 \le \text{min\_n} \le \text{max\_n} \le 3$).
2. **Stopword Elimination:** Evaluates candidate edges against a comprehensive, zero-dependency English stopword set. Phrases beginning or ending with stopwords are pruned.
3. **Dense Semantic Embedding:** Uses `sentence-transformers/all-MiniLM-L6-v2` (~22.7M parameters, ~90 MB memory) to generate 384-dimensional dense vectors with mean-pooling and L2 unit-normalization.
4. **Cosine Similarity Ranking:** Calculates semantic similarity between the document vector and candidate vectors $\text{sim}(\mathbf{d}, \mathbf{c}) = \frac{\mathbf{d} \cdot \mathbf{c}}{\|\mathbf{d}\| \|\mathbf{c}\|}$, sorting results in descending relevance order.

### 2. Abstractive Summarization & Long-Document Chunking
1. **Sequence-to-Sequence Generation:** Leverages `sshleifer/distilbart-cnn-12-6` (~306M parameters, ~600 MB VRAM / ~1.2 GB RAM), fine-tuned on CNN/DailyMail for concise abstractive summarization.
2. **Long-Document Strategy:** When input texts exceed the single-pass chunk limit (`SUMMARIZATION_CHUNK_SIZE=1800` characters), the document is split across sentence boundaries using `chunk_text_by_sentences`. Each chunk is summarized independently, followed by a bounded hierarchical consolidation loop across chunk summaries.
3. **Usage Attribution:** Metering tracks `cost_units` equal to the number of document chunks processed, ensuring fair billing attribution.

### 3. Hardware Requirements & Model Lifecycle
- **Target Hardware:** Tested and validated for modern multicore CPUs and laptop GPUs (e.g. NVIDIA RTX 4050 6 GB VRAM, 16 GB RAM).
- **LRU Cache Protection:** With `MODEL_MAX_CACHED=3`, loading subsequent models automatically evicts the least recently used idle model while shielding active inferences.
- **Known Limitations:**
  - *Information Compression:* Abstractive summarization compresses text, which can omit minor contextual nuances in highly technical documents.
  - *Hallucination Risk:* As with all pretrained autoregressive sequence-to-sequence models, extreme length constraints or domain-specific jargon may produce mild phrasing hallucinations.

---

## Generative NLP & Structured Information Extraction (Milestone 6)

### 1. Local LLM Architecture via Ollama
NLPulse integrates local generative LLMs via an asynchronous HTTP client communicating with the local Ollama daemon service (`/api/generate`):
- **Decoupled Architecture:** Ollama runs as a standalone local service without adding heavy generative dependencies to the core API runtime. Offline test suites run independently with zero live network or daemon dependencies.
- **Model Recommendation:** Configured by default for lightweight instruct models (e.g. `qwen2.5:3b` or `llama3.2:3b`), requiring under 2.5 GB VRAM and operating smoothly on mobile/consumer workstation GPUs (such as RTX 4050 6 GB VRAM) as well as CPU fallbacks.
- **Resilient Error Classification:** Custom exception hierarchy mapping service states:
  - `503 Service Unavailable`: Local Ollama daemon unreachable or not running.
  - `404 Not Found`: Model tag not downloaded in Ollama (`ollama pull <model>`).
  - `504 Gateway Timeout`: Upstream Ollama request timed out before completing generation.
  - `502 Bad Gateway`: Upstream Ollama generation failure or malformed response.
  - `422 Unprocessable Entity`: Model output violates user-provided JSON schema.

### 2. Schema-Grounded Information Extraction
- **Flexible Schema Input:** Supports standard JSON Schema specifications (`schema_definition`) as well as developer-friendly field lists (`fields: [{"name": ..., "type": ...}]`).
- **Strict Anti-Hallucination Grounding:** System instructions enforce strict factual adherence to the source document. If a requested property is not explicitly stated or directly inferable, the model is instructed to output JSON `null`.
- **Programmatic Post-Generation Validation:** Strips any markdown code block wrappers, parses raw JSON, validates expected property types (numbers, booleans, arrays), and normalizes placeholder strings (e.g. "N/A", "none", "unknown") to `null`.
- **Transparent Token-Based Metering:** Cost units are calculated proportionally based on input prompt and generated completion tokens (`EXTRACTION_COST_PER_1K_TOKENS=5`), recording atomic usage records linked to the developer tenant.

---

## Batch Processing Architecture (Milestone 7)

### 1. Execution Strategy: Tensor Batching vs. Bounded Concurrency
NLPulse implements task-appropriate batching strategies tailored to 6 GB VRAM / 16 GB RAM resource limits and M3 `ModelRegistry` lifecycle safety:

| Capability | `execution_mode` | Strategy |
| :--- | :--- | :--- |
| **Sentiment** | `batched_inference` | True micro-batched tensor forward passes (`SentimentModel.predict_batch`, chunked by `BATCH_INFERENCE_CHUNK_SIZE`) with single-item fallback to isolate failing inputs |
| **NER** | `batched_inference` | Micro-batched token classification (`NERModel.predict_batch`) with subword aggregation and isolated fallback |
| **Zero-Shot Classification** | `grouped_batched_inference` | Groups items sharing `(candidate_labels, multi_label)` into micro-batched NLI passes via `ZeroShotClassificationModel.predict_batch` |
| **Keyword Extraction** | `deduplicated_batched_embeddings` | Extracts candidate n-grams per item, deduplicates all documents + candidates into a shared embedding pool, and encodes in bounded PyTorch tensor batches |
| **Summarization** | `bounded_concurrent` | Fixed worker pool (`min(N, BATCH_MAX_CONCURRENCY)`) executing hierarchical chunked summarization per document on the shared `SummarizationModel` |
| **Structured Extraction** | `bounded_concurrent` | Fixed worker pool (`min(N, BATCH_MAX_CONCURRENCY, BATCH_EXTRACTION_CONCURRENCY)`) running Ollama `/api/generate` calls with per-item schema grounding |

### 2. Batch Configuration
Configure resource and throughput bounds via `.env` or environment variables:

```env
BATCH_MAX_SIZE=20                 # Max items per batch request (422 above this limit)
BATCH_MAX_CONCURRENCY=4           # Max concurrent worker coroutines per batch
BATCH_INFERENCE_CHUNK_SIZE=8      # Micro-batch size for Transformer tensor passes
BATCH_EXTRACTION_CONCURRENCY=2    # Concurrency cap for local Ollama extraction requests
```

### 3. API Examples & Partial Failure Behavior
Request example (`POST /api/v1/nlp/batch/sentiment`):

```json
{
  "items": [
    {"id": "review-1", "text": "NLPulse delivers fast and accurate inference."},
    {"id": "review-2", "text": "   "},
    {"id": "review-3", "text": "Unexpected timeout during downstream test."}
  ]
}
```

Response (`200 OK`, preserving exact item order and `id`):

```json
{
  "capability": "sentiment",
  "execution_mode": "batched_inference",
  "model_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  "total_items": 3,
  "succeeded_count": 2,
  "failed_count": 1,
  "total_cost_units": 2,
  "latency_ms": 14.2,
  "results": [
    {
      "index": 0,
      "id": "review-1",
      "status": "success",
      "result": {
        "label": "POSITIVE",
        "confidence": 0.9921,
        "scores": {"POSITIVE": 0.9921, "NEGATIVE": 0.0079},
        "model_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "latency_ms": 6.8
      },
      "error": null,
      "cost_units": 1,
      "latency_ms": 6.8
    },
    {
      "index": 1,
      "id": "review-2",
      "status": "error",
      "result": null,
      "error": {
        "code": "validation_error",
        "message": "Input text cannot be empty or solely whitespace.",
        "status_code": 422
      },
      "cost_units": 0,
      "latency_ms": 0.0
    },
    {
      "index": 2,
      "id": "review-3",
      "status": "success",
      "result": {
        "label": "NEGATIVE",
        "confidence": 0.9634,
        "scores": {"NEGATIVE": 0.9634, "POSITIVE": 0.0366},
        "model_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "latency_ms": 6.8
      },
      "error": null,
      "cost_units": 1,
      "latency_ms": 6.8
    }
  ]
}
```

### 4. Failure Semantics, Atomic Quota Reservation & Metering
- **Envelope vs. Item Validation:** Empty (`items: []`) or oversized (`len(items) > BATCH_MAX_SIZE`) requests fail with HTTP `422`. Invalid individual items (blank text, invalid parameter ranges, malformed per-item extraction output, or upstream Ollama errors) fail safely inside `results[i].error` without invalidating healthy items.
- **Atomic Quota Reservation:** `UsageService.reserve_quota` serializes per-tenant reservation checks for both single requests (`reserve_single_quota`) and batches using `QuotaReservationManager` without holding long-lived PostgreSQL row locks across slow inference, preventing both concurrent TOCTOU quota overruns and PostgreSQL AB-BA deadlocks. Committed usage records atomically settle and release reservations under the same developer lock.
- **Accurate Usage Settlement:** Only items with `status == "success"` generate persisted `UsageRecord` rows (`endpoint="/api/v1/nlp/batch/<capability>"`) and contribute to `total_cost_units`. Failed or cancelled items incur zero quota consumption and zero cost units, and reservations are always released in `finally` blocks even on `asyncio.CancelledError`.

---

## Getting Started

### Prerequisites

- **Python 3.12+**
- **Docker Desktop** (optional for local PostgreSQL 16 container)

### 1. Setup Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment configuration:

```bash
cp .env.example .env
```

### 3. Start Database (Optional)

```bash
docker compose up -d
```

### 4. Run Development Server

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive documentation:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Running Tests

Run the offline unit and integration test suite with Pytest (smoke tests requiring live Hugging Face downloads are deselected by default in `pyproject.toml`):

```bash
pytest -v
```

To opt into real PostgreSQL 16 integration tests (`@pytest.mark.postgres`):

```bash
NLPULSE_TEST_POSTGRES_URL="postgresql+asyncpg://user:password@localhost:5432/db" pytest -m postgres -v
```

To opt into live Hugging Face model smoke tests (requires internet access and model downloads):

```bash
RUN_REAL_MODEL_SMOKE=1 pytest -m smoke -v
```

### Deployment & Verification Scope (M8)

- **Single-Worker Coordination Boundary:** `SlidingWindowRateLimiter` (`src/core/rate_limiter.py`) and `QuotaReservationManager` (`src/services/usage_service.py`) store synchronization state in process memory. Single-worker `uvicorn src.main:app` execution enforces strict sliding-window rate limits and atomic deadlock-free quota reservations; multi-worker deployment (`--workers > 1`) requires adding an external shared state backend.
- **Verification Environment Notes:** Alembic migrations, constraints, cascades, and concurrent HTTP batch + single request flows were verified against real PostgreSQL 16.15 (WSL) and SQLite offline fixtures; live Hugging Face downloads, local CUDA kernel execution (CPU-only PyTorch installed in the development environment), and a live Ollama daemon remain optional/unverified runtime targets outside the standard offline suite.

---

## Roadmap (M1–M9)

| Milestone | Focus | Main Deliverable | Status |
| :--- | :--- | :--- | :--- |
| **M1** | Foundations & Database | Project structure, PostgreSQL, SQLAlchemy, models, migrations | ✅ Complete |
| **M2** | Security & Authentication | API keys, authentication, scopes, rate limiting, quotas | ✅ Complete |
| **M3** | Model Lifecycle Management | Model loading, caching, GPU/CPU management, concurrency | ✅ Complete |
| **M4** | Core NLP | Sentiment analysis, NER, zero-shot classification | ✅ Complete |
| **M5** | Advanced NLP | Keyword extraction, summarization | ✅ Complete |
| **M6** | Generative NLP | Ollama integration, structured information extraction | ✅ Complete |
| **M7** | Batch Processing | Micro-batched inference, bounded worker pools, quota reservation, partial failures | ✅ Complete |
| **M8** | Integration & Verification | API integration, end-to-end tests, security & PostgreSQL verification | ✅ Complete |
| **M9** | Documentation & Portfolio | README, architecture docs, ADRs, interview preparation, final release | ⏳ Next |
