# ADR-002: Async SQLAlchemy 2.x and PostgreSQL 16 Architecture

- **Status:** Accepted (Current)

## Context

Every authenticated API request queries PostgreSQL for API key hash verification, active developer status, monthly usage aggregation, and post-inference `UsageRecord` persistence — all within an asynchronous FastAPI event loop handling slow HTTP (Ollama) and thread-offloaded ML workloads concurrently.

## Decision

Use **SQLAlchemy 2.x asynchronous ORM** (`AsyncEngine`, `async_sessionmaker`, `AsyncSession`) paired with the pure-Python/Cython **`asyncpg`** driver against **PostgreSQL 16**, managed via async **Alembic** migrations (`migrations/env.py`):
- Request-scoped `AsyncSession` instances (`expire_on_commit=False`, `autoflush=False`) via `get_db()` in `src/db/session.py`.
- `pool_pre_ping=True`, `pool_size=10`, `max_overflow=20` when connecting to PostgreSQL (`NullPool` when SQLite is configured for tests).
- Explicit SQL `CHECK` constraints, `ON DELETE CASCADE` (`developers -> api_keys, usage_records`), and `ON DELETE SET NULL` (`api_keys -> usage_records`) defined in ORM models and mirrored identically in Alembic revision `1687e8d000b1`.

## Alternatives Considered

1. **Synchronous SQLAlchemy (`psycopg2` / `psycopg`) inside FastAPI:** Easier setup, but database I/O would block worker threads or the event loop under concurrent loads unless every endpoint dropped to sync FastAPI threadpools.
2. **Raw `asyncpg` SQL queries without an ORM:** Minimal runtime overhead, but sacrifices declarative schema management, Alembic `autogenerate` drift checking (`alembic check`), and portable `aiosqlite` integration testing.

## Consequences

- **Positive:** Non-blocking PostgreSQL I/O under concurrent HTTP workloads, automated migration drift detection (`alembic check`), strong database-enforced invariants (`status_code`, non-negative quotas/latencies/cost units), and rapid offline CI/local testing using `aiosqlite`.
- **Negative:** SQLite lacks exact PostgreSQL MVCC locking and transaction timing; PostgreSQL behavior must still be explicitly verified via `tests/integration/test_postgres_real.py` (`@pytest.mark.postgres`). Rolled-back ORM instances expire attributes under async sessions and cannot be lazily refreshed outside greenlet contexts.

## Current Status

Implemented in `src/db/`, `src/models/`, and `migrations/`. Verified against real PostgreSQL 16.15 via WSL (`alembic upgrade head`, `alembic check`, constraints, cascades, and concurrency suites).
