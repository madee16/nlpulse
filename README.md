# NLPulse — Intelligent Text Analysis API (NLP-as-a-Service)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-e92063.svg)](https://docs.pydantic.dev/)
[![PostgreSQL 16](https://img.shields.io/badge/PostgreSQL-16-336791.svg?logo=postgresql)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NLPulse** is a high-performance developer-facing REST API providing Natural Language Processing as a Service. It delivers end-to-end NLP pipelines including sentiment analysis, named entity recognition (NER), zero-shot text classification, keyword extraction, text summarization, and LLM-powered structured extraction.

---

## Current Status: M1 · Task 1.2

- ✅ **M1 · Task 1.1:** Scaffolding, configuration, logging, health route, and Docker environment
- ✅ **M1 · Task 1.2:** Database architecture, SQLAlchemy 2.x async session lifecycle, multi-tenant relational schema (`Developer`, `APIKey`, `UsageRecord`), Alembic async migrations, and tenant isolation tests (22/22 tests passing)

---

## Tech Stack

- **Language:** Python 3.12
- **Framework:** FastAPI
- **Configuration & Validation:** Pydantic v2, Pydantic Settings
- **Database:** PostgreSQL 16 (orchestrated via Docker Compose)
- **Database Driver / ORM:** SQLAlchemy 2.x (async), Asyncpg
- **Testing:** Pytest, pytest-asyncio, HTTPX
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
│   ├── env.py                # Async migration runner
│   ├── script.py.mako
│   └── versions/             # Migration scripts
├── src/
│   ├── __init__.py           # Application package root
│   ├── main.py               # FastAPI application entry point & lifespan
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py           # Dependency injection (get_db)
│   │   └── routes/
│   │       ├── __init__.py
│   │       └── health.py     # Health check endpoint (/api/v1/health)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py         # Pydantic Settings & environment variables
│   │   ├── logging.py        # Structured JSON & console logging
│   │   └── security.py       # API key hashing (SHA-256) & constant-time verification
│   ├── db/
│   │   ├── __init__.py
│   │   ├── base.py           # Declarative base & metadata registry
│   │   └── session.py        # Async engine & sessionmaker
│   └── models/
│       ├── __init__.py       # Model exports
│       ├── api_key.py        # APIKey model & hash indexing
│       ├── developer.py      # Developer tenant root model
│       └── usage_record.py   # UsageRecord metering & audit model
└── tests/
    ├── __init__.py
    ├── conftest.py           # Pytest fixtures (async client & in-memory test DB)
    ├── integration/
    │   ├── __init__.py
    │   └── test_migrations.py # Alembic upgrade/downgrade integration tests
    └── unit/
        ├── __init__.py
        ├── test_config.py    # Configuration tests
        ├── test_health.py    # Health check endpoint tests
        ├── test_logging.py   # Structured logging tests
        ├── test_models.py    # Model relationships, constraints, cascades
        ├── test_security.py  # Cryptographic API key tests
        └── test_tenant_isolation.py # Multi-tenant isolation tests
```

---

## Getting Started

### Prerequisites

- **Python 3.12+**
- **Docker Desktop** (optional for local database container)

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

### 3. Start Database (Optional for Task 1.1)

```bash
docker compose up -d
```

### 4. Run Development Server

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Once running, access:
- **Interactive API Documentation (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative Documentation (ReDoc):** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **API Root:** [http://localhost:8000/](http://localhost:8000/)
- **Health Check:** [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

## Running Tests

Run the full test suite with Pytest:

```bash
pytest -v
```

Expected output:
```text
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
tests/unit/test_models.py::test_delete_api_key_preserves_usage_record PASSED
tests/unit/test_models.py::test_database_check_constraints PASSED
tests/unit/test_security.py::test_generate_api_key PASSED
tests/unit/test_security.py::test_verify_api_key PASSED
tests/unit/test_tenant_isolation.py::test_tenant_isolation_api_keys PASSED
tests/unit/test_tenant_isolation.py::test_tenant_isolation_usage_records PASSED
```

---

## Roadmap

- [x] **M1 · Task 1.1:** Project Scaffolding & Development Environment
- [x] **M1 · Task 1.2:** Database Architecture, Migrations & Tenancy
- [ ] **M1 · Task 1.3:** API Security, Authentication & Rate Limiting
- [ ] **M2 · Task 2.1:** Sentiment Analysis Engine
- [ ] **M2 · Task 2.2:** Named Entity Recognition (NER) Pipeline
- [ ] **M2 · Task 2.3:** Zero-Shot Classification & Keyword Extraction
- [ ] **M3 · Task 3.1:** Text Summarization Service
- [ ] **M3 · Task 3.2:** LLM Structured Extraction (Ollama Integration)
- [ ] **M3 · Task 3.3:** Batch Processing & Async Worker Integration
