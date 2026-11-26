# NLPulse — Intelligent Text Analysis API

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-e92063.svg)](https://docs.pydantic.dev/)
[![PostgreSQL 16](https://img.shields.io/badge/PostgreSQL-16-336791.svg?logo=postgresql)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NLPulse** is a developer-facing **NLP-as-a-Service REST API** built with FastAPI, PostgreSQL 16, Hugging Face Transformers, and optional local Ollama inference. It combines applied NLP pipelines, local LLM-based structured extraction, and batch processing with API key authentication, granular scope authorization, rate limiting, atomic quota enforcement, and bounded model memory management.

> **Documentation Index:** [Architecture](docs/ARCHITECTURE.md) · [ADRs](docs/adr/001-api-first-monolithic-architecture.md) · [Interview Guide](docs/INTERVIEW_GUIDE.md) · [Portfolio Overview](docs/PORTFOLIO_OVERVIEW.md)

---

## Why NLPulse Exists

Production AI backends rarely wrap a single inference call in isolation. Even small workstation deployments (`6 GB VRAM / 16 GB RAM`) require controlled lazy model loading, cache eviction, worker-thread isolation, tenant quotas, safe error boundaries, and partial-failure batch semantics. NLPulse addresses this by providing an API-first monolithic service with standardized REST contracts across both deterministic Transformer models and optional local generative extraction.

---

## Core Capabilities

NLPulse separates **deterministic local Transformer/embedding pipelines** (requiring only Python dependencies and Hugging Face weights on demand) from **optional local generative extraction** (delegated asynchronously to a local Ollama service):

| Capability | Individual Endpoint | Batch Endpoint | Execution Approach |
| :--- | :--- | :--- | :--- |
| **Sentiment Analysis** | `POST /api/v1/nlp/sentiment` | `POST /api/v1/nlp/batch/sentiment` | Transformer sequence classification; micro-batched tensor forward passes in batch mode |
| **Named Entity Recognition** | `POST /api/v1/nlp/ner` | `POST /api/v1/nlp/batch/ner` | Token classification with subword aggregation and character offsets; micro-batched passes in batch mode |
| **Zero-Shot Classification** | `POST /api/v1/nlp/classification` | `POST /api/v1/nlp/batch/classification` | NLI zero-shot scoring (single-label softmax or multi-label); grouped by label set in batch mode |
| **Keyword Extraction** | `POST /api/v1/nlp/keywords` | `POST /api/v1/nlp/batch/keywords` | L2-normalized dense embeddings + cosine similarity ranking; cross-document candidate deduplication in batch mode |
| **Abstractive Summarization** | `POST /api/v1/nlp/summarization` | `POST /api/v1/nlp/batch/summarization` | Token-bounded sentence chunking and hierarchical consolidation; bounded concurrent worker pool in batch mode |
| **Structured Extraction *(Optional Ollama)*** | `POST /api/v1/nlp/extraction` | `POST /api/v1/nlp/batch/extraction` | Grounded JSON extraction against JSON Schema or field specs via local Ollama `/api/generate` (`httpx`) |

---

## Technology Stack

| Layer | Technologies Used | Role in NLPulse |
| :--- | :--- | :--- |
| **API & Validation** | Python 3.12, FastAPI, Uvicorn, Pydantic v2, Pydantic Settings, `email-validator`, `jsonschema` | Asynchronous REST routing, OpenAPI generation, security headers/CORS middleware, strict request/response, Draft 2020-12 extraction schema & env validation |
| **Local Playground *(Optional)*** | React 19, Vite, TypeScript | Developer-facing local UI for health, model inspection, and single-request capability testing (`frontend/`) |
| **Persistence & Migrations** | PostgreSQL 16, SQLAlchemy 2.x (async), `asyncpg`, Alembic, Docker Compose | Async ORM sessions (`pool_pre_ping`), tenant/key/usage schema with check constraints, cascade/audit foreign keys, reproducible schema migrations |
| **Traditional & Embedding NLP** | PyTorch, Hugging Face `transformers`, NumPy | Lazy-loaded sequence/token classification, NLI zero-shot scoring, mean-pooled dense sentence embeddings, seq2seq summarization |
| **Generative Extraction** | `httpx` (`AsyncClient`), Ollama *(optional)* | Decoupled HTTP transport to local LLM daemon (`POST /api/generate`) with timeout and error classification |
| **Testing** | Pytest, `pytest-asyncio`, `httpx` (`ASGITransport`), `aiosqlite` | Fast offline SQLite + mocked pipeline suites alongside opt-in PostgreSQL 16 and opt-in Hugging Face smoke suites |

---

## Architecture Overview

```mermaid
flowchart LR
    Client["Client Application"] --> Middleware["SecurityHeadersMiddleware and CORSMiddleware"]
    Middleware --> Deps["Auth, Scope Check, Rate Limit, Quota Guard"]
    Deps --> Services["NLPService, BatchService, ExtractionService, UsageService"]

    Deps --> RateLimiter["SlidingWindowRateLimiter (process-local)"]
    Services --> QuotaMgr["QuotaReservationManager (process-local)"]
    Services --> DB[("PostgreSQL 16")]

    Services --> Registry["ModelRegistry (LRU max_cached_models=3)"]
    Registry --> Workers["BaseModel + asyncio.to_thread Workers (CPU / CUDA auto-detect)"]
    Services --> Ollama["OllamaClient (Optional Local LLM)"]
```

Key design highlights:
- **Deadlock-Free Quota & Usage Metering:** Monthly quota checks reserve capacity before inference using 64-stripe in-process locks (`QuotaReservationManager`) instead of holding PostgreSQL `SELECT ... FOR UPDATE` row locks across slow ML calls, settling usage records atomically under the same lock (`src/services/usage_service.py`).
- **Lifecycle-Safe Model Cache:** `ModelRegistry` deduplicates concurrent loads per model, enforces `MODEL_MAX_CACHED=3` across parallel distinct loads, protects busy models from LRU eviction, and shields off-loop worker threads (`asyncio.shield`) so inference timeouts never trigger unsafe unloading (`src/core/models/base.py`, `src/services/model_registry.py`).
- **Tenant-Isolated Authentication:** `X-API-Key` secrets (`nlp_live_...`) are verified via SHA-256 digests (`hmac.compare_digest` utilities in `src/core/security.py`), enforcing non-delegable scopes, free-tier registration limits, and safe error sanitization (`503` on SQLAlchemy errors, `500` fallback).

For full sequence diagrams, database constraints, and architectural tradeoffs, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/adr/](docs/adr/001-api-first-monolithic-architecture.md).

---

## Repository Layout

| Path | Responsibility |
| :--- | :--- |
| `src/api/` | FastAPI dependency guards (`deps.py`) and routers (`health`, `developers`, `keys`, `usage`, `models`, `nlp`, `batch`, `protected`) |
| `src/core/` | Settings, device/CUDA detection, domain exceptions, structured JSON/text logging, rate limiter, crypto helpers, NLP utils, and `BaseModel`/HF model wrappers |
| `src/db/` & `src/models/` | Async SQLAlchemy engine/session factory and ORM models (`Developer`, `APIKey`, `UsageRecord`) |
| `src/schemas/` | Pydantic v2 request/response contracts for auth, usage, telemetry, individual NLP, extraction, and batch envelopes |
| `src/services/` | Domain services (`NLPService`, `BatchService`, `ExtractionService`, `ModelRegistry`, `UsageService`, `APIKeyService`, `DeveloperService`, `OllamaClient`) |
| `migrations/` | Async Alembic environment (`env.py`) and schema revision history (`1687e8d000b1_initial_schema.py`) |
| `tests/` | Unit suites, HTTP/SQLite integration suites, security regressions, opt-in PostgreSQL 16 integration (`test_postgres_real.py`), and opt-in HF smoke tests (`test_real_models_smoke.py`) |
| `frontend/` | Optional React + Vite + TypeScript local NLP playground (`sessionStorage` API key, zero mocked inference) |
| `docs/` | Deep system architecture, ADRs (`001`–`007`), technical interview guide, and portfolio overview |

---

## Quick Start

### Prerequisites
- **Required:** Python **3.12+**, **PostgreSQL 16** (via Docker Compose or local PostgreSQL installation).
- **Optional:** **Ollama** (only if calling `/api/v1/nlp/extraction` or `/api/v1/nlp/batch/extraction`), CUDA-enabled PyTorch + NVIDIA GPU (CPU execution works by default), internet access (first-time Hugging Face model download on demand).

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/madee16/nlpulse.git
cd nlpulse

python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# Install runtime + test dependencies (or `pip install -e ".[dev]"`)
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Adjust `POSTGRES_*` or `DATABASE_URL` in `.env` if not using the default Docker Compose settings. Default startup does not require Ollama.

### 3. Start PostgreSQL 16 and Run Migrations

```bash
docker compose up -d
alembic upgrade head
```

*(If running PostgreSQL natively without Docker Compose, ensure the target database exists and `DATABASE_URL` points to it before running `alembic upgrade head`.)*

### 4. Start the API Server (Single-Worker)

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Interactive OpenAPI explorers:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

### 5. Optional: Enable Local Generative Extraction (Ollama)

```bash
ollama pull qwen2.5:3b
ollama serve
```

If Ollama is not running, all other NLP, batch, auth, and model routes continue working normally; extraction routes cleanly report `503` (or per-item `503` in batch mode).

---

## Local Frontend / Playground

An optional lightweight developer interface lives in `frontend/` (`React 19 + Vite + TypeScript`) for testing the real FastAPI backend interactively without modifying API contracts.

### Run Locally

1. Start PostgreSQL and the backend with Vite's local origin allowed (already included in `.env.example` `CORS_ORIGINS`):

```bash
alembic upgrade head
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

2. In a second terminal, start the Vite development server:

```bash
cd frontend
npm install
npm run dev
```

3. Open `http://localhost:5173`, confirm backend health in the header bar, paste an `X-API-Key` issued by `POST /api/v1/developers`, and run requests across all six NLP capabilities or inspect `/api/v1/models`, `/api/v1/usage/summary`, and `/api/v1/hardware` (requires `admin` scope).

- **Session-only key storage:** `X-API-Key` is kept strictly in browser `sessionStorage` and cleared immediately via **Clear Key** (never persisted to `localStorage` or logged).
- **Real backend responses:** Every structured panel and Raw JSON view displays actual HTTP responses, `Retry-After` headers, and error details directly from FastAPI without mocked outputs.

### Frontend Verification Commands

```bash
cd frontend
npm run typecheck
npm test
npm run build
```

---

## Configuration Reference

Settings are defined in `src/core/config.py` and templated in `.env.example`:

| Variable | Purpose | Required | Default | Sensitive |
| :--- | :--- | :--- | :--- | :--- |
| `APP_NAME` | Service display title | No | `NLPulse` | No |
| `APP_ENV` | Environment label (`development`, `staging`, etc.) | No | `development` | No |
| `APP_VERSION` | Semantic API version exposed in OpenAPI/health | No | `0.1.0` | No |
| `DEBUG` | Enables SQLAlchemy SQL echo logging when `true` | No | `false` | No |
| `API_V1_PREFIX` | URL prefix for versioned API routes | No | `/api/v1` | No |
| `HOST` | Bind host configuration field | No | `0.0.0.0` | No |
| `PORT` | Bind port configuration field | No | `8000` | No |
| `LOG_LEVEL` | Root logging threshold (`INFO`, `DEBUG`, etc.) | No | `INFO` | No |
| `LOG_FORMAT` | Output format (`json` or `text`) | No | `json` | No |
| `POSTGRES_SERVER` | PostgreSQL hostname when assembling `DATABASE_URL` | No* | `localhost` | No |
| `POSTGRES_PORT` | PostgreSQL port | No* | `5432` | No |
| `POSTGRES_USER` | PostgreSQL username | No* | `postgres` | No |
| `POSTGRES_PASSWORD` | PostgreSQL password (`assemble_db_connection`) | No* | `postgrespassword` | **Yes** |
| `POSTGRES_DB` | PostgreSQL database name | No* | `nlpulse` | No |
| `DATABASE_URL` | Full async SQLAlchemy URL (`postgresql+asyncpg://...`) | No* | Derived from `POSTGRES_*` | **Yes** |
| `CORS_ORIGINS` | Allowed CORS origins (JSON list or comma-separated) | No | `["*"]` (`.env.example` sets localhost origins) | No |
| `MODEL_DEVICE_PREFERENCE` | Compute target (`auto`, `cuda`, `cpu`) | No | `auto` | No |
| `MODEL_MAX_CACHED` | Maximum concurrently cached HF models (`>= 1`) | No | `3` | No |
| `MODEL_DEFAULT_CONCURRENCY` | Per-model semaphore concurrency cap | No | `2` | No |
| `MODEL_INFERENCE_TIMEOUT_SECONDS` | Per-inference thread wait timeout (seconds) | No | `30.0` | No |
| `MODEL_ENABLE_WARMUP` | Reserved setting; **not currently implemented** (startup always registers models lazily) | No | `false` | No |
| `MODEL_SENTIMENT_ID` | Hugging Face sentiment pipeline repo ID | No | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | No |
| `MODEL_NER_ID` | Hugging Face NER pipeline repo ID | No | `dslim/bert-base-NER` | No |
| `MODEL_ZERO_SHOT_ID` | Hugging Face zero-shot NLI pipeline repo ID | No | `valhalla/distilbart-mnli-12-1` | No |
| `NLP_MAX_TEXT_LENGTH` | Max input chars for sentiment, NER, zero-shot, keywords | No | `5000` | No |
| `MODEL_EMBEDDING_ID` | Hugging Face embedding model repo ID | No | `sentence-transformers/all-MiniLM-L6-v2` | No |
| `MODEL_SUMMARIZATION_ID` | Hugging Face summarization pipeline repo ID | No | `sshleifer/distilbart-cnn-12-6` | No |
| `KEYWORD_MAX_CANDIDATES` | Maximum extracted n-gram candidates per item | No | `100` | No |
| `SUMMARIZATION_CHUNK_SIZE` | Character target per summarization chunk (with 600-token cap) | No | `1800` | No |
| `OLLAMA_BASE_URL` | Base URL for optional local Ollama HTTP API | No | `http://localhost:11434` | No |
| `OLLAMA_MODEL` | Default local Ollama model tag | No | `qwen2.5:3b` | No |
| `OLLAMA_TIMEOUT_SECONDS` | HTTP read/request timeout for Ollama generation (s) | No | `60.0` | No |
| `OLLAMA_MAX_INPUT_LENGTH` | Max source text length for extraction requests | No | `15000` | No |
| `EXTRACTION_COST_PER_1K_TOKENS` | Cost-unit weight per 1,000 Ollama tokens | No | `5` | No |
| `BATCH_MAX_SIZE` | Max items allowed in any batch envelope | No | `20` | No |
| `BATCH_MAX_CONCURRENCY` | Max concurrent worker coroutines per batch | No | `4` | No |
| `BATCH_INFERENCE_CHUNK_SIZE` | Micro-batch size for batched Transformer passes | No | `8` | No |
| `BATCH_EXTRACTION_CONCURRENCY` | Max concurrent Ollama workers per batch | No | `2` | No |
| `REGISTRATION_RATE_LIMIT_REQUESTS` | Max public `POST /api/v1/developers` attempts per window per IP | No | `5` | No |
| `REGISTRATION_RATE_LIMIT_WINDOW_SECONDS` | Sliding window duration in seconds for registration admission limit | No | `60` | No |
| `TRUSTED_PROXY_IPS` | Explicit proxy IPs/CIDRs permitted to supply right-most `X-Forwarded-For` | No | `[]` | No |
| `NLPULSE_TEST_POSTGRES_URL` | Test-only env var enabling `pytest -m postgres` | No | Unset | **Yes** |
| `RUN_REAL_MODEL_SMOKE` | Test-only flag (`1`) enabling `pytest -m smoke` | No | Unset | No |

*\*Production PostgreSQL deployments should override default credentials via `DATABASE_URL` or `POSTGRES_PASSWORD`.*

---

## API Overview & Examples

### Endpoint Summary

All routes register cleanly in OpenAPI (`/api/v1/openapi.json`, `26` paths total). Protected endpoints expect the `X-API-Key` header.

| Method | Path | Auth | Accepted Scopes | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/`, `/health`, `/api/v1/health` | None | - | Discovery metadata and service health |
| `POST` | `/api/v1/developers` | None | - | Register developer (free tier, `1000` quota) and return bootstrap `X-API-Key` once |
| `GET` | `/api/v1/developers/me` | `X-API-Key` | Any active key | Inspect authenticated developer profile and monthly quota |
| `POST` | `/api/v1/keys` | `X-API-Key` | `keys:write`, `admin`, `*` | Issue child API key (non-empty subset of caller scopes) |
| `GET` | `/api/v1/keys` | `X-API-Key` | `keys:read`, `keys:write`, `admin`, `*` | List API keys owned by tenant (hashes never exposed) |
| `DELETE` | `/api/v1/keys/{key_id}` | `X-API-Key` | `keys:write`, `admin`, `*` | Revoke an owned API key |
| `GET` | `/api/v1/usage/summary` | `X-API-Key` | `usage:read`, `admin`, `*` | Monthly calendar usage, quota remaining, and percentage |
| `POST` | `/api/v1/nlp/sentiment` | `X-API-Key` | `sentiment:read`, `nlp:read`, `analyze:read`, `admin`, `*` | Analyze text sentiment polarity and score distribution |
| `POST` | `/api/v1/nlp/ner` | `X-API-Key` | `ner:read`, `nlp:read`, `analyze:read`, `admin`, `*` | Extract named entity spans with character offsets |
| `POST` | `/api/v1/nlp/classification` | `X-API-Key` | `classification:read`, `nlp:read`, `analyze:read`, `admin`, `*` | Zero-shot classification against candidate labels |
| `POST` | `/api/v1/nlp/keywords` | `X-API-Key` | `keywords:read`, `nlp:read`, `analyze:read`, `admin`, `*` | Dense embedding keyword and keyphrase ranking |
| `POST` | `/api/v1/nlp/summarization` | `X-API-Key` | `summarization:read`, `nlp:read`, `analyze:read`, `admin`, `*` | Chunked abstractive summarization with length controls |
| `POST` | `/api/v1/nlp/extraction` | `X-API-Key` | `extraction:read`, `generate:read`, `nlp:read`, `analyze:read`, `admin`, `*` | Schema-validated extraction via local Ollama |
| `POST` | `/api/v1/nlp/batch/{capability}` | `X-API-Key` | Same scope rule as matching single capability | Batch `sentiment`, `ner`, `classification`, `keywords`, `summarization`, `extraction` |
| `GET` | `/api/v1/models` | `X-API-Key` | `models:read`, `analyze:read`, `admin`, `*` | List registered models and lifecycle status |
| `GET` | `/api/v1/models/{model_id:path}` | `X-API-Key` | `models:read`, `analyze:read`, `admin`, `*` | Inspect telemetry for a model ID (supports HF `org/model` paths) |
| `POST` | `/api/v1/models/{model_id:path}/unload` | `X-API-Key` | `admin`, `*` | Unload a cached model from memory |
| `GET` | `/api/v1/hardware` | `X-API-Key` | `admin`, `*` | Inspect host CPU/CUDA hardware metrics |
| `GET` | `/api/v1/protected/sentiment` | `X-API-Key` | `sentiment:read`, `analyze:read`, `admin`, `*` | Minimal metered auth/scope verification route |
| `GET` | `/api/v1/protected/admin-only` | `X-API-Key` | `admin`, `*` | Minimal admin-scope verification route |

### Representative Schema-Verified Examples

<details>
<summary><strong>1. Sentiment Analysis — POST /api/v1/nlp/sentiment</strong></summary>

Request:
```json
{
  "text": "NLPulse delivers consistent inference latency with clear validation."
}
```
Response (`200 OK`):
```json
{
  "label": "POSITIVE",
  "confidence": 0.9994,
  "scores": {
    "POSITIVE": 0.9994,
    "NEGATIVE": 0.0006
  },
  "model_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  "latency_ms": 28.4
}
```
</details>

<details>
<summary><strong>2. Named Entity Recognition — POST /api/v1/nlp/ner</strong></summary>

Request:
```json
{
  "text": "Sundar Pichai announced new infrastructure in Mountain View."
}
```
Response (`200 OK`):
```json
{
  "entities": [
    {
      "entity": "PER",
      "text": "Sundar Pichai",
      "confidence": 0.9981,
      "start_offset": 0,
      "end_offset": 13
    },
    {
      "entity": "LOC",
      "text": "Mountain View",
      "confidence": 0.9965,
      "start_offset": 46,
      "end_offset": 59
    }
  ],
  "entity_count": 2,
  "model_id": "dslim/bert-base-NER",
  "latency_ms": 34.1
}
```
</details>

<details>
<summary><strong>3. Zero-Shot Classification — POST /api/v1/nlp/classification</strong></summary>

Request:
```json
{
  "text": "Apple reported record quarterly revenue driven by services.",
  "candidate_labels": ["finance", "technology", "sports"],
  "multi_label": false
}
```
Response (`200 OK`):
```json
{
  "top_label": "finance",
  "top_score": 0.6124,
  "predictions": [
    {"label": "finance", "score": 0.6124},
    {"label": "technology", "score": 0.3712},
    {"label": "sports", "score": 0.0164}
  ],
  "multi_label": false,
  "model_id": "valhalla/distilbart-mnli-12-1",
  "latency_ms": 74.0
}
```
</details>

<details>
<summary><strong>4. Keyword Extraction — POST /api/v1/nlp/keywords</strong></summary>

Request:
```json
{
  "text": "FastAPI provides automatic OpenAPI documentation and high asynchronous throughput.",
  "max_keywords": 3,
  "ngram_range": [1, 2]
}
```
Response (`200 OK`):
```json
{
  "keywords": [
    {"keyword": "OpenAPI documentation", "score": 0.6842},
    {"keyword": "FastAPI", "score": 0.6195},
    {"keyword": "asynchronous throughput", "score": 0.5903}
  ],
  "total_candidates": 14,
  "model_id": "sentence-transformers/all-MiniLM-L6-v2",
  "latency_ms": 19.7
}
```
</details>

<details>
<summary><strong>5. Abstractive Summarization — POST /api/v1/nlp/summarization</strong></summary>

Request:
```json
{
  "text": "PostgreSQL 16 introduces performance improvements in query parallelism and logical replication. Teams running async Python services benefit from indexed time-range usage logs and explicit foreign key cascades.",
  "min_length": 15,
  "max_length": 60,
  "do_sample": false
}
```
Response (`200 OK`):
```json
{
  "summary": "PostgreSQL 16 improves query parallelism and logical replication for async Python workloads using indexed logs and explicit cascades.",
  "original_character_count": 203,
  "summary_character_count": 131,
  "chunks_processed": 1,
  "model_id": "sshleifer/distilbart-cnn-12-6",
  "latency_ms": 312.5
}
```
</details>

<details>
<summary><strong>6. Structured Extraction (Ollama) — POST /api/v1/nlp/extraction</strong></summary>

Request:
```json
{
  "text": "Apple Inc. announced Q4 revenue of $89.5 billion in Cupertino.",
  "fields": [
    {"name": "company", "type": "string", "required": true},
    {"name": "revenue", "type": "string", "required": true},
    {"name": "ceo", "type": "string", "required": false}
  ],
  "temperature": 0.0
}
```
Response (`200 OK`):
```json
{
  "extracted_data": {
    "company": "Apple Inc.",
    "revenue": "$89.5 billion",
    "ceo": null
  },
  "model": "qwen2.5:3b",
  "prompt_tokens": 118,
  "completion_tokens": 24,
  "total_tokens": 142,
  "cost_units": 1,
  "latency_ms": 485.0
}
```
</details>

<details>
<summary><strong>7. Batch Sentiment with Partial Failure — POST /api/v1/nlp/batch/sentiment</strong></summary>

Request:
```json
{
  "items": [
    {"id": "doc-1", "text": "NLPulse delivers consistent inference latency."},
    {"id": "doc-2", "text": "   "}
  ]
}
```
Response (`200 OK`):
```json
{
  "capability": "sentiment",
  "execution_mode": "batched_inference",
  "model_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  "total_items": 2,
  "succeeded_count": 1,
  "failed_count": 1,
  "total_cost_units": 1,
  "latency_ms": 14.2,
  "results": [
    {
      "index": 0,
      "id": "doc-1",
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
      "id": "doc-2",
      "status": "error",
      "result": null,
      "error": {
        "code": "validation_error",
        "message": "Input text cannot be empty or solely whitespace.",
        "status_code": 422
      },
      "cost_units": 0,
      "latency_ms": 0.0
    }
  ]
}
```
</details>

---

## Authentication, Security & Deployment Boundary

- **API Key Lifecycle:** `POST /api/v1/developers` issues a `nlp_live_...` secret returned once (`32` URL-safe bytes). Only a 16-character prefix and SHA-256 hex digest live in `api_keys`. Child keys (`POST /api/v1/keys`) require explicit non-empty scopes (`min_length=1`) capped to a subset of the caller's scopes; public registration always enforces `tier="free"` (`1000` quota) without `admin`.
- **Public Registration Admission Control:** `POST /api/v1/developers` applies process-local per-IP sliding-window rate limiting (`REGISTRATION_RATE_LIMIT_REQUESTS=5` per `60s`, returning `429` + `Retry-After` before any DB write). Direct peer IP (`request.client.host`) is enforced unless it matches configured `TRUSTED_PROXY_IPS` (in which case right-most untrusted `X-Forwarded-For` traversal prevents client spoofing). This does **not** provide distributed multi-node abuse protection.
- **Structured Extraction Validation & Redaction:** Caller `schema_definition` payloads validate pre- and post-inference against **JSON Schema Draft 2020-12** (`jsonschema.Draft202012Validator` with built-in `FormatChecker` for `date`, `time`, `email`, `idn-email`, `idn-hostname`, `ipv4`, `ipv6`, `regex`, `uuid`). Missing `required` fields, invalid primitive types, bounds/pattern/format/enum violations raise `422` without inventing defaults (`fields` mode compiles nullable `[type, "null"]` types). Malformed/invalid LLM output and document content are never included in application logs or validation error details.
- **Legacy Scope Compatibility:** Granular scopes (`sentiment:read`, `ner:read`, `classification:read`, `keywords:read`, `summarization:read`, `extraction:read`) coexist with `nlp:read`, `analyze:read`, `generate:read` (extraction), `models:read`, `keys:read`, `keys:write`, `usage:read`, `admin`, and `*`.
- **Rate Limiting & Atomic Quota:** `SlidingWindowRateLimiter` allows `60` requests per `60` seconds per `api_key.key_hash` (`64` lock stripes, `429` + `Retry-After`). Monthly quotas reserve units up front (`reserve_single_quota` or `BatchService`) and release atomically upon database settlement (or via `asyncio.shield` cleanup).
- **Single-Worker Deployment Boundary:** `rate_limiter`, `registration_rate_limiter`, and `QuotaReservationManager` store state in **process memory**. Running single-worker `uvicorn src.main:app` enforces configured limits with zero PostgreSQL deadlocks; multi-worker deployment (`--workers > 1`) is **not supported** without an external shared state store (see [ADR-007](docs/adr/007-single-worker-deployment-boundary.md)).

---

## Model Management & Supported Defaults

All Hugging Face models start `unloaded`, transition through `loading` -> `ready` on demand, evict the least-recently-used idle model beyond `MODEL_MAX_CACHED=3`, and auto-detect CUDA vs. CPU via `DeviceManager` (`MODEL_DEVICE_PREFERENCE=auto`). Note that hardware detection telemetry is implemented and unit-tested, while PyTorch installed in the current environment is CPU-only (`cuda_available=False`).

| Task | Default Hugging Face Repo ID | Upstream License | Est. VRAM / RAM (`main.py`) |
| :--- | :--- | :--- | :--- |
| Sentiment Analysis | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | `apache-2.0` | `270 MB` / `350 MB` |
| Named Entity Recognition | `dslim/bert-base-NER` | `mit` | `420 MB` / `500 MB` |
| Zero-Shot Classification | `valhalla/distilbart-mnli-12-1` | Derived from `facebook/bart-large-mnli` (`mit`; no explicit HF YAML tag) | `550 MB` / `700 MB` |
| Embeddings / Keywords | `sentence-transformers/all-MiniLM-L6-v2` | `apache-2.0` | `90 MB` / `120 MB` |
| Summarization | `sshleifer/distilbart-cnn-12-6` | `apache-2.0` | `600 MB` / `1200 MB` |

Local generative extraction defaults to Ollama model tag `qwen2.5:3b` via an external Ollama service and never consumes slots in `ModelRegistry`.

---

## Testing & Verification Tiers

NLPulse explicitly separates fast offline tests (SQLite `aiosqlite` + mocked Hugging Face/Ollama pipelines) from real external services:

| Test Tier | Command | Prerequisites | What It Proves |
| :--- | :--- | :--- | :--- |
| **Default Offline Suite** | `pytest -v` | None (uses `aiosqlite` and mocks) | API routing, schemas, registration admission control, JSON Schema extraction & redaction, quotas, LRU eviction, batch partial failures (`-m "not smoke"` default in `pyproject.toml`) |
| **Real PostgreSQL 16 Suite** | `NLPULSE_TEST_POSTGRES_URL="postgresql+asyncpg://user:pass@localhost:5432/nlpulse_test" pytest -m postgres -v` | Running disposable PostgreSQL 16 DB (`_test` / `test_` suffix required) | Reproducible Alembic `upgrade head`, `alembic check`, `downgrade base` reset, `CHECK` constraints, `CASCADE` / `SET NULL`, and concurrent batch + single locking |
| **Live HF Smoke Suite** | `RUN_REAL_MODEL_SMOKE=1 pytest -m smoke -v` | Internet access + HF model download cache | End-to-end weight downloads and real CPU/CUDA forward passes for Sentiment, NER, and Zero-Shot |
| **Frontend Playground Suite** | `cd frontend && npm test && npm run typecheck && npm run build` | Node.js / npm | API transport, `sessionStorage` key isolation, capability tab & error rendering, TypeScript compilation, Vite production bundle |

*(Default `pytest` skips `@pytest.mark.postgres` unless `NLPULSE_TEST_POSTGRES_URL` is exported and deselects `@pytest.mark.smoke`.)*

---

## Troubleshooting

- **PowerShell activation blocked:** Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` before `.\.venv\Scripts\Activate.ps1`.
- **PostgreSQL connection refused during migrations/startup:** Confirm PostgreSQL 16 is running (`docker compose ps` or local service) and matches `DATABASE_URL` in `.env`.
- **`503 Local Ollama service unavailable` on extraction routes:** Expected when Ollama is stopped; start `ollama serve` and run `ollama pull qwen2.5:3b` only if using `/api/v1/nlp/extraction` or `/api/v1/nlp/batch/extraction`.
- **CUDA unavailable (`cuda_available=false` on `/api/v1/hardware`):** Check whether CPU-only PyTorch wheels are installed (`python -c "import torch; print(torch.cuda.is_available())"`) and ensure `MODEL_DEVICE_PREFERENCE` is `auto` or `cpu`.

---

## Limitations & Future Improvements

Current verified limitations:
1. **Single-Worker Coordination:** API-key rate limiting (`rate_limiter`), public registration rate limiting (`registration_rate_limiter`), and in-flight quota tracking (`QuotaReservationManager`) are process-local (`--workers > 1` unsupported and not distributed abuse protection).
2. **Unverified Physical CUDA Execution:** While `DeviceManager` implements CUDA checks and OOM cleanup (tested via mocks), `.venv` carries CPU-only PyTorch, leaving real GPU kernel execution unverified.
3. **Optional External Services:** Live Hugging Face inference (`-m smoke`), end-to-end Ollama text generation (probing local Ollama returned `404` because `qwen2.5:3b` was not pulled), and running PostgreSQL inside Docker Desktop remain optional/environment-dependent (PostgreSQL 16 migrations and concurrency were verified natively via WSL PostgreSQL 16.15).
4. **JSON Schema Format Scope:** Built-in `jsonschema.FormatChecker()` validates `date`, `time`, `email`, `idn-email`, `idn-hostname`, `ipv4`, `ipv6`, `regex`, and `uuid` without external format validator plugins (`date-time`/`uri` format extras not bundled).
5. **Reserved Config Flag:** `MODEL_ENABLE_WARMUP` exists in `Settings` / `.env.example` as a reserved flag and does not currently preload models at startup.

Potential future enhancements (not yet implemented): Redis-backed distributed rate limiting and quota leases for horizontal scaling, GPU-stage CI smoke execution, `/ready` warmup lifecycle support, and asynchronous queued batch jobs (`202 Accepted`) for large document workloads.
