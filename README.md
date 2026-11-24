# NLPulse — Intelligent Text Analysis API (NLP-as-a-Service)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-e92063.svg)](https://docs.pydantic.dev/)
[![PostgreSQL 16](https://img.shields.io/badge/PostgreSQL-16-336791.svg?logo=postgresql)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NLPulse** is a high-performance developer-facing REST API providing Natural Language Processing as a Service. It delivers end-to-end NLP pipelines including sentiment analysis, named entity recognition (NER), zero-shot text classification, keyword extraction, text summarization, and LLM-powered structured extraction.

---

## Current Status: M1 & M2 Complete ✅

- ✅ **M1: Foundations & Database** — Project structure, PostgreSQL 16, async SQLAlchemy 2.x session lifecycle, models (`Developer`, `APIKey`, `UsageRecord`), and Alembic async migrations.
- ✅ **M2: Security & Authentication** — API keys, SHA-256 constant-time hashing, scope enforcement, sliding window rate limiting, monthly database quotas, and tenant isolation (39/39 tests passing).
- ⏳ **M3: Model Lifecycle Management** — Ready for kickoff (Model loading, caching, GPU/CPU management, concurrency).

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
│   │   ├── exceptions.py     # Domain exceptions (Auth, Quota, RateLimit, Conflict)
│   │   ├── logging.py        # Structured JSON & console logging
│   │   ├── rate_limiter.py   # Thread-safe in-memory sliding window rate limiter
│   │   └── security.py       # SHA-256 API key hashing & constant-time comparison
│   ├── db/
│   │   ├── __init__.py
│   │   ├── base.py           # Declarative base & metadata registry
│   │   └── session.py        # Async engine & sessionmaker
│   ├── models/
│   │   ├── __init__.py       # Model exports
│   │   ├── api_key.py        # APIKey model & hash indexing
│   │   ├── developer.py      # Developer tenant root model
│   │   └── usage_record.py   # UsageRecord metering & audit model
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── api_key.py        # API key request/response schemas
│   │   ├── developer.py      # Developer registration/profile schemas
│   │   └── usage.py          # Usage summary schemas
│   └── services/
│       ├── __init__.py
│       ├── api_key_service.py   # Key issuance, hashing, and auth
│       ├── developer_service.py # Tenant management & quota assignment
│       └── usage_service.py     # Metering & calendar month quota checks
└── tests/
    ├── __init__.py
    ├── conftest.py           # Pytest fixtures (async client & in-memory test DB)
    ├── integration/
    │   ├── __init__.py
    │   ├── test_auth_api.py   # End-to-end API registration, auth, and scopes
    │   └── test_migrations.py # Alembic upgrade/downgrade integration tests
    └── unit/
        ├── __init__.py
        ├── test_config.py     # Configuration tests
        ├── test_health.py     # Health check endpoint tests
        ├── test_logging.py    # Structured logging tests
        ├── test_models.py     # Model relationships, constraints, cascades
        ├── test_rate_limiter.py # Sliding window rate limiter tests
        ├── test_security.py   # Cryptographic API key tests
        ├── test_services.py   # Business services unit tests
        └── test_tenant_isolation.py # Multi-tenant isolation tests
```

---

## API Endpoints (Milestone 1)

| Method | Endpoint | Auth Required | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | No | API root & discovery metadata |
| `GET` | `/api/v1/health` | No | Service health check |
| `POST` | `/api/v1/developers` | No | Register new developer account |
| `GET` | `/api/v1/developers/me` | `X-API-Key` | View authenticated developer profile & quota |
| `POST` | `/api/v1/developers/{id}/keys` | No | Bootstrap initial API key for developer |
| `POST` | `/api/v1/keys` | `X-API-Key` | Create a new API key (plain key returned once) |
| `GET` | `/api/v1/keys` | `X-API-Key` | List all API keys for tenant |
| `DELETE` | `/api/v1/keys/{key_id}` | `X-API-Key` | Revoke an API key |
| `GET` | `/api/v1/usage/summary` | `X-API-Key` | Monthly quota consumption & remaining requests |
| `GET` | `/api/v1/protected/sentiment` | `X-API-Key` | Protected route requiring `sentiment:read` |
| `GET` | `/api/v1/protected/admin-only` | `X-API-Key` | Protected route requiring `admin` |

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
============================= 39 passed in 3.72s =============================
```

---

## Roadmap (M1–M9)

| Milestone | Focus | Main Deliverable | Status |
| :--- | :--- | :--- | :--- |
| **M1** | Foundations & Database | Project structure, PostgreSQL, SQLAlchemy, models, migrations | ✅ Complete |
| **M2** | Security & Authentication | API keys, authentication, scopes, rate limiting, quotas | ✅ Complete |
| **M3** | Model Lifecycle Management | Model loading, caching, GPU/CPU management, concurrency | ⏳ Next |
| **M4** | Core NLP | Sentiment analysis, NER, zero-shot classification | 📋 Planned |
| **M5** | Advanced NLP | Keyword extraction, summarization | 📋 Planned |
| **M6** | Generative NLP | Ollama integration, structured information extraction | 📋 Planned |
| **M7** | Batch Processing | Efficient multi-text inference and benchmarking | 📋 Planned |
| **M8** | Integration & Verification | API integration, end-to-end tests, security verification | 📋 Planned |
| **M9** | Documentation & Portfolio | README, architecture docs, ADRs, interview preparation, final release | 📋 Planned |
