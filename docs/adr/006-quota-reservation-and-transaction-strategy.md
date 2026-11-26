# ADR-006: Quota Reservation and Transaction Strategy

- **Status:** Accepted (Supersedes initial `SELECT ... FOR UPDATE` design after M8 PostgreSQL 16 verification)

## Context

Each developer tenant has a monthly request/cost-unit quota checked before executing slow ML or Ollama inference, while usage is committed only on success (`UsageService.record_usage` / `UsageService.settle_batch_usage`).

During PostgreSQL 16 concurrency testing, holding a database `SELECT ... FOR UPDATE` row lock across slow inference while settlement also acquired the in-process `QuotaReservationManager` lock caused an **AB-BA deadlock** between concurrent batch requests. Additionally, single NLP requests previously used non-reserving `check_quota`, leaving a TOCTOU race window under parallel single requests, and naive `finally:` release created a brief double-counting window between DB `commit()` and FastAPI teardown.

## Decision

Adopt an atomic in-process lock + DB commit handoff strategy in `UsageService` (`src/services/usage_service.py`) and `reserve_single_quota` (`src/api/deps.py`):
1. **Reserve Before Inference (No DB Row Locks Held):** Both single (`reserve_single_quota`, `units=1`) and batch (`BatchService._run_with_quota_guard`, `units=valid_count`) acquire the developer's striped `QuotaReservationManager` asyncio lock, read committed DB usage + in-flight `_reserved[developer_id]`, reject with `429` if `used + reserved + units > monthly_request_quota`, increment `_reserved`, and release the lock before slow ML/LLM execution begins.
2. **Atomic In-Lock Settlement:** On success, both `UsageService.record_usage` (via `session.info["_quota_reservations"]` guarded by `isinstance(session_info, dict)`) and `UsageService.settle_batch_usage` commit `UsageRecord` row(s) and decrement `_reserved` **inside the same `quota_reservation_manager.get_lock(developer_id)` critical section**, eliminating any double-counting window.
3. **Shielded Failure Release:** If inference fails or a client cancels before settlement, `finally:` blocks release unused reserved units inside `asyncio.shield(...)`.

## Alternatives Considered

1. **PostgreSQL `SELECT ... FOR UPDATE` held across inference:** Prevents TOCTOU across processes, but pins database connections and row locks for seconds of GPU/CPU inference and deadlocks with in-process locks or concurrent transactions.
2. **Optimistic DB insert before inference with compensating delete on failure:** Avoids reservations, but violates strict success-only `UsageRecord` semantics and complicates aborted partial failures.

## Consequences

- **Positive:** Eliminates PostgreSQL AB-BA deadlocks, closes both single-request and batch TOCTOU quota races cleanly under real PostgreSQL 16 MVCC concurrency, removes double-counting glitches, and preserves fast connection turnover.
- **Negative:** Relies on `QuotaReservationManager` living in one worker process; multi-worker deployment would allow separate workers to see uncommitted/unreserved concurrent in-flight work until DB settlement commits.

## Current Status

Implemented and verified on both SQLite and live WSL PostgreSQL 16.15 (`tests/integration/test_postgres_real.py` confirms zero deadlock and exact `10/10` quota enforcement across mixed concurrent batches and single requests).
