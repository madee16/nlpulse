# NLPulse — Intelligent Text Analysis API (NLP-as-a-Service)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-e92063.svg)](https://docs.pydantic.dev/)
[![PostgreSQL 16](https://img.shields.io/badge/PostgreSQL-16-336791.svg?logo=postgresql)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NLPulse** is a high-performance developer-facing REST API providing Natural Language Processing as a Service. It delivers end-to-end NLP pipelines including sentiment analysis, named entity recognition (NER), zero-shot text classification, keyword extraction, text summarization, and LLM-powered structured extraction.

---

## Current Status: M1, M2, M3, M4, M5 & M6 Complete ✅

- ✅ **M1: Foundations & Database** — Project structure, PostgreSQL 16, async SQLAlchemy 2.x session lifecycle, models (`Developer`, `APIKey`, `UsageRecord`), and Alembic async migrations.
- ✅ **M2: Security & Authentication** — API keys, SHA-256 constant-time hashing, scope enforcement, sliding window rate limiting, monthly database quotas, and tenant isolation.
- ✅ **M3: Model Lifecycle Management** — Lazy model initialization, thread-safe `ModelRegistry`, LRU eviction with in-use protection, hardware detection (`DeviceManager` for CUDA GPU/CPU), per-model concurrency control, event-loop offloading via `asyncio.to_thread` with shielded thread lifecycle retention, and FastAPI lifespan integration.
- ✅ **M4: Core NLP** — Sentiment analysis (`distilbert-base-uncased-finetuned-sst-2-english`), Named Entity Recognition (`bert-base-NER` with subword BPE token aggregation), and Zero-Shot text classification (`distilbart-mnli-12-1` with single-label softmax & multi-label sigmoid scoring). Complete end-to-end API integration with rate limiting, monthly quota enforcement, and per-tenant usage metering.
- ✅ **M5: Advanced NLP** — Dense embedding keyword and keyphrase extraction (`sentence-transformers/all-MiniLM-L6-v2` with cosine similarity ranking and customizable n-gram / stopword filtering) and abstractive document summarization (`sshleifer/distilbart-cnn-12-6` with token-bounded sentence chunking and hierarchical consolidation). Fully integrated with LRU cache eviction, scopes (`keywords:read`, `summarization:read`), rate limits, quotas, and atomic usage metering.
- ✅ **M6: Generative NLP** — Local LLM integration via asynchronous `OllamaClient`, structured information extraction (`POST /api/v1/nlp/extraction`) adhering to JSON Schema or field specifications, programmatic schema validation with null fallbacks for unstated fields, strict factual grounding, configurable instruct models (e.g. `qwen2.5:3b`, `llama3.2:3b`), and transparent token-based cost unit metering.
- ⏳ **M7: Batch Processing** — Next: Efficient multi-text inference and benchmarking.

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
│   │       ├── developers.py # Developer registration & profile endpoints
│   │       ├── health.py     # Health check endpoint (/api/v1/health)
│   │       ├── keys.py       # API key issuance, listing, and revocation
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
│   │   ├── developer.py      # Developer registration/profile schemas
│   │   ├── model.py          # Model telemetry & hardware metrics schemas
│   │   ├── nlp.py            # NLP requests, responses, and entity schemas
│   │   ├── extraction.py     # Schema definitions & extraction schemas (M6)
│   │   └── usage.py          # Usage summary schemas
│   └── services/
│       ├── __init__.py
│       ├── api_key_service.py   # Key issuance, hashing, and auth
│       ├── developer_service.py # Tenant management & quota assignment
│       ├── extraction_service.py# Generative structured extraction service (M6)
│       ├── model_registry.py    # Lazy loading, LRU caching, eviction & locks
│       ├── nlp_service.py       # Core NLP orchestration & atomic usage metering
│       ├── ollama_client.py     # Asynchronous local Ollama HTTP client (M6)
│       └── usage_service.py     # Metering & calendar month quota checks
└── tests/
    ├── __init__.py
    ├── conftest.py           # Pytest fixtures (async client & in-memory test DB)
    ├── integration/
    │   ├── __init__.py
    │   ├── test_auth_api.py   # End-to-end API registration, auth, and scopes
    │   ├── test_extraction_api.py # M6 structured extraction API tests
    │   ├── test_m5_nlp_api.py # Keywords and summarization API tests
    │   ├── test_migrations.py # Alembic upgrade/downgrade integration tests
    │   ├── test_models_api.py # Model registry and hardware telemetry API tests
    │   └── test_nlp_api.py    # Core NLP (sentiment, ner, classification) API tests
    └── unit/
        ├── __init__.py
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

## API Endpoints (Milestones 1–6)

| Method | Endpoint | Auth Required | Scopes | Description |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/` | No | - | API root & discovery metadata |
| `GET` | `/api/v1/health` | No | - | Service health check |
| `POST` | `/api/v1/developers` | No | - | Register developer (enforces free tier & returns bootstrap API key) |
| `GET` | `/api/v1/developers/me` | `X-API-Key` | - | View authenticated developer profile & quota |
| `POST` | `/api/v1/keys` | `X-API-Key` | - | Create a child API key (scopes must be subset of caller's key) |
| `GET` | `/api/v1/keys` | `X-API-Key` | - | List all API keys for tenant |
| `DELETE` | `/api/v1/keys/{key_id}` | `X-API-Key` | - | Revoke an API key |
| `GET` | `/api/v1/usage/summary` | `X-API-Key` | `usage:read` | Monthly quota consumption & remaining requests |
| `POST` | `/api/v1/nlp/sentiment` | `X-API-Key` | `sentiment:read` | DistilBERT sentiment analysis & confidence distribution |
| `POST` | `/api/v1/nlp/ner` | `X-API-Key` | `ner:read` | BERT named entity recognition with character span offsets |
| `POST` | `/api/v1/nlp/classification` | `X-API-Key` | `classification:read` | DistilBART zero-shot classification (single or multi-label) |
| `POST` | `/api/v1/nlp/keywords` | `X-API-Key` | `keywords:read` | MiniLM dense embedding keyword & keyphrase ranking |
| `POST` | `/api/v1/nlp/summarization` | `X-API-Key` | `summarization:read` | DistilBART abstractive text summarization with chunking |
| `POST` | `/api/v1/nlp/extraction` | `X-API-Key` | `extraction:read` | Local LLM grounded structured information extraction (M6) |
| `GET` | `/api/v1/models` | `X-API-Key` | `models:read` / `admin` | List registered models and operational status |
| `GET` | `/api/v1/models/{model_id}` | `X-API-Key` | `models:read` / `admin` | Retrieve model telemetry and loading state |
| `POST` | `/api/v1/models/{model_id}/unload` | `X-API-Key` | `admin` | Explicitly unload model and release GPU/RAM |
| `GET` | `/api/v1/hardware` | `X-API-Key` | `admin` | Inspect compute hardware, GPU availability, and VRAM |

---

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
2. **Long-Document Strategy:** When input texts exceed the single-pass token limit (`SUMMARIZATION_CHUNK_SIZE=2500` characters), the document is split across sentence boundaries using `chunk_text_by_sentences`. Each chunk is summarized independently, followed by a consolidation pass across chunk summaries.
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
  - `502 Bad Gateway`: Upstream Ollama timeout or generation failure.
  - `422 Unprocessable Entity`: Model output violates user-provided JSON schema.

### 2. Schema-Grounded Information Extraction
- **Flexible Schema Input:** Supports standard JSON Schema specifications (`schema_definition`) as well as developer-friendly field lists (`fields: [{"name": ..., "type": ...}]`).
- **Strict Anti-Hallucination Grounding:** System instructions enforce strict factual adherence to the source document. If a requested property is not explicitly stated or directly inferable, the model is instructed to output JSON `null`.
- **Programmatic Post-Generation Validation:** Strips any markdown code block wrappers, parses raw JSON, validates expected property types (numbers, booleans, arrays), and normalizes placeholder strings (e.g. "N/A", "none", "unknown") to `null`.
- **Transparent Token-Based Metering:** Cost units are calculated proportionally based on input prompt and generated completion tokens (`EXTRACTION_COST_PER_1K_TOKENS=5`), recording atomic usage records linked to the developer tenant.

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

Run the full test suite with Pytest:

```bash
pytest -v
```

Expected output:
```text
tests/integration/test_auth_api.py::test_developer_registration PASSED
tests/integration/test_auth_api.py::test_developer_registration_duplicate_email PASSED
tests/integration/test_auth_api.py::test_api_key_issuance_and_auth_flow PASSED
tests/integration/test_auth_api.py::test_authentication_failures PASSED
tests/integration/test_auth_api.py::test_scope_enforcement PASSED
tests/integration/test_auth_api.py::test_usage_summary_endpoint PASSED
tests/integration/test_auth_api.py::test_tenant_cannot_revoke_other_tenant_key PASSED
tests/integration/test_auth_api.py::test_rate_limiting_http PASSED
tests/integration/test_auth_api.py::test_quota_exhaustion_http PASSED
tests/integration/test_migrations.py::test_alembic_migration_upgrade_and_downgrade PASSED
tests/unit/test_config.py::test_default_settings PASSED
tests/unit/test_config.py::test_settings_environment_override PASSED
tests/unit/test_config.py::test_get_settings_caching PASSED
tests/unit/test_health.py::test_health_check_endpoint PASSED
tests/unit/test_health.py::test_root_health_endpoint PASSED
tests/unit/test_health.py::test_root_endpoint PASSED
tests/unit/test_logging.py::test_json_formatter PASSED
tests/unit/test_logging.py::test_setup_logging_text_and_json PASSED
tests/unit/test_models.py::test_create_developer PASSED
tests/unit/test_models.py::test_developer_unique_email PASSED
tests/unit/test_models.py::test_create_api_key_and_relationship PASSED
tests/unit/test_models.py::test_api_key_unique_hash PASSED
tests/unit/test_models.py::test_api_key_revocation_and_expiration PASSED
tests/unit/test_models.py::test_create_usage_record PASSED
tests/unit/test_models.py::test_cascade_delete_developer PASSED
tests/unit/test_delete_api_key_preserves_usage_record PASSED
tests/unit/test_models.py::test_database_check_constraints PASSED
tests/unit/test_rate_limiter.py::test_rate_limiter_allows_under_limit PASSED
tests/unit/test_rate_limiter.py::test_rate_limiter_blocks_over_limit PASSED
tests/unit/test_rate_limiter.py::test_rate_limiter_resets_after_window PASSED
tests/unit/test_rate_limiter.py::test_rate_limiter_independent_keys PASSED
tests/unit/test_security.py::test_generate_api_key PASSED
tests/unit/test_security.py::test_verify_api_key PASSED
tests/unit/test_services.py::test_developer_service_crud PASSED
tests/unit/test_services.py::test_api_key_service_lifecycle PASSED
tests/unit/test_services.py::test_api_key_expired_rejection PASSED
tests/unit/test_services.py::test_usage_service_quota_enforcement PASSED
tests/unit/test_tenant_isolation.py::test_tenant_isolation_api_keys PASSED
tests/unit/test_tenant_isolation.py::test_tenant_isolation_usage_records PASSED
====================== 133 passed, 3 deselected in 9.45s ======================
```

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
| **M7** | Batch Processing | Efficient multi-text inference and benchmarking | ⏳ Next |
| **M8** | Integration & Verification | API integration, end-to-end tests, security verification | 📋 Planned |
| **M9** | Documentation & Portfolio | README, architecture docs, ADRs, interview preparation, final release | 📋 Planned |
