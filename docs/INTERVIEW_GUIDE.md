# NLPulse — Technical Interview Guide

This guide prepares you to explain **NLPulse** clearly, concretely, and honestly in AI Engineering and backend technical interviews. Every explanation maps directly to actual code in `src/`, `migrations/`, and `tests/`.

---

## 1. Project Elevator Pitches

### 30-Second Pitch
"NLPulse is a developer-facing NLP-as-a-Service REST platform built with Python 3.12, FastAPI, PostgreSQL 16, Hugging Face Transformers, and optional local Ollama inference. It exposes six core text-analysis capabilities — sentiment, NER, zero-shot classification, semantic keyword extraction, chunked abstractive summarization, and schema-validated generative extraction — alongside batch endpoints, API key authentication, sliding-window rate limiting, and deadlock-free monthly quota enforcement."

### 2-Minute Technical Summary
"I built NLPulse to combine applied NLP pipelines with production-grade API engineering on constrained workstation hardware (`6 GB VRAM / 16 GB RAM`). On the ML side, it loads five Transformer and Sentence-Transformer models lazily through a centralized `ModelRegistry` with per-model loading locks, bounded semaphores (`max_concurrency=2`), worker-thread offloading via `asyncio.to_thread`, and LRU eviction when resident models hit `MODEL_MAX_CACHED=3`. Alongside Hugging Face pipelines, it integrates an optional local Ollama daemon for JSON-Schema-grounded structured extraction with recursive validation and fallback normalization.

On the backend side, every API key uses a `secrets.token_urlsafe(32)` secret stored only as a SHA-256 hash, enforcing non-delegable granular scopes, striped sliding-window rate limits, and atomic monthly quota reservations. Batch endpoints combine vectorized PyTorch micro-batching (Sentiment, NER, grouped Zero-Shot), cross-document deduplicated candidate embeddings (Keywords), and bounded async worker pools (Summarization, Ollama Extraction) with per-item failure isolation and success-only usage metering."

### Architecture Walkthrough (5 Minutes)
Walk through [docs/ARCHITECTURE.md](./ARCHITECTURE.md) in request order:
1. **Security middleware & error translation** (`src/main.py`) -> security headers on all HTTP responses and sanitized `503`/`500` handling.
2. **Auth & resource guards** (`src/api/deps.py`) -> `X-API-Key` SHA-256 lookup (`APIKeyService`), `require_scopes`, striped `SlidingWindowRateLimiter` on `api_key.key_hash`, and atomic `QuotaReservationManager`.
3. **Single vs. Batch orchestration** (`NLPService`, `ExtractionService`, `BatchService`) -> upfront input validation vs. per-item `BatchItemResult` isolation, vectorized chunking, and strict success-only `UsageRecord` settlement.
4. **Model lifecycle & PostgreSQL persistence** (`ModelRegistry`, `BaseModel`, `UsageService`) -> slot-reserved LRU caching, shielded worker thread tracking so timeouts never trigger unsafe mid-flight unloads, and single-worker coordination boundaries (see [ADR-006](./adr/006-quota-reservation-and-transaction-strategy.md) & [ADR-007](./adr/007-single-worker-deployment-boundary.md)).

---

## 2. Technical Questions & Model Answers

### Backend, FastAPI & Database

**Q1: How does NLPulse prevent synchronous PyTorch inference from blocking FastAPI's async event loop?**  
Every `BaseModel` executes inference inside `_run_worker()` (`src/core/models/base.py`), which acquires an `asyncio.Semaphore(max_concurrency)` (default `2`), offloads the synchronous forward pass with `asyncio.to_thread(...)`, and awaits `asyncio.shield(worker_task)` under `timeout_seconds` (`30.0s`). Database I/O and Ollama HTTP calls remain non-blocking on the main event loop.

**Q2: What happens if a PyTorch worker thread times out?**  
Cancelling `asyncio.wait_for()` does **not** terminate a Python OS thread. If the semaphore or `_active_inferences` counter were released in an `except asyncio.TimeoutError` block while the thread kept running on GPU/RAM, `ModelRegistry` could attempt to unload or overload an active model. NLPulse shields `worker_task` and attaches `worker_task.add_done_callback(_cleanup_worker)` so `_active_inferences` and `_semaphore` release strictly when the OS thread completes.

**Q3: Why does `src/db/session.py` configure `expire_on_commit=False` and `NullPool` conditionally?**  
With `AsyncSession`, expired attributes trigger synchronous-style lazy loads on access, raising `MissingGreenlet` outside explicit async loaders. `expire_on_commit=False` prevents committed ORM objects (e.g., `Developer`, `APIKey`) from expiring across `commit()` inside request dependencies. For PostgreSQL, connection pooling uses `pool_size=10`, `max_overflow=20`, and `pool_pre_ping=True`; for SQLite test URLs (`sqlite+...`), `NullPool` is selected because `aiosqlite` does not use standard multi-connection pooling. (Note: `await db.rollback()` still expires attributes, which we handled explicitly in `test_postgres_real.py` by capturing `dev_id` before rollback.)

**Q4: How do foreign keys behave when an API key is revoked/deleted vs. when a developer account is deleted?**  
`APIKey.developer_id` and `UsageRecord.developer_id` use `ON DELETE CASCADE`, removing a tenant cleanly deletes its keys and usage history. `UsageRecord.api_key_id` is nullable with `ON DELETE SET NULL`, preserving historical quota metering and audit records even if a key is deleted (`src/models/usage_record.py`).

### Security, Rate Limiting & Quotas

**Q5: How are API keys stored, verified, and scoped?**  
`generate_api_key("nlp_live")` generates 32 bytes of `secrets.token_urlsafe(32)` entropy (`src/core/security.py`). Only the first 16 chars (`key_prefix`) and the 64-char SHA-256 hex digest (`key_hash`) are stored; raw keys return strictly once on creation. Verification hashes the incoming `X-API-Key`, looks up `key_hash`, checks `is_active`, `revoked_at`, and `expires_at`, and updates `last_used_at`. Endpoints check scopes via `require_scopes(...)`: `admin` or `*` satisfies everything; otherwise at least one granted scope must match. Child key creation prevents privilege escalation by verifying `requested_scopes <= authorizing_scopes`, rejecting empty scopes (`min_length=1`), and restricting public `POST /api/v1/developers` registration to `"tier": "free"` (`1000` monthly quota) without `admin`. Note `analyze:read` remains accepted on NLP/model routes for backward compatibility.

**Q6: How is tenant isolation enforced?**  
All key listing/revocation (`APIKeyService`), profile lookup (`GET /api/v1/developers/me`), and usage queries (`UsageService`) filter on `developer_id` derived from the verified `APIKey`, never from caller-supplied path or query IDs. Calling `DELETE /api/v1/keys/{other_tenant_key_id}` returns `404` (`ResourceNotFoundError`).

**Q7: How does the rate limiter work and avoid lock contention or memory leaks?**  
`SlidingWindowRateLimiter` (`src/core/rate_limiter.py`) tracks request timestamps in a 60-second sliding window (`default_max_requests=60`) keyed by `api_key.key_hash`. It uses 64 striped `asyncio.Lock` instances (`abs(hash(key)) % 64`) so distinct keys do not serialize each other, prunes timestamps outside the window on every check, and evicts stale keys via `prune_stale_entries()` when tracked keys reach `max_tracked_keys=10000`. Exceeded limits raise HTTP `429` with `Retry-After`.

**Q8: Why use atomic in-memory quota reservations instead of just counting `UsageRecord` rows?**  
Checking `SELECT count/sum` alone creates a Time-Of-Check-to-Time-Of-Use (TOCTOU) race when concurrent slow NLP/LLM requests arrive simultaneously before any request finishes and inserts a `UsageRecord`. `QuotaReservationManager` reserves `units` (`1` for single requests, `valid_count` for batches) inside a 64-stripe per-tenant `asyncio.Lock` **before** inference starts, and atomically commits `UsageRecord`(s) and releases the reservation **inside the same lock** on completion (or in an `asyncio.shield` `finally` block on error/cancellation).

### Applied NLP & Local LLM Pipelines

**Q9: How does `NERModel` compute accurate character offsets across subword tokens?**  
`NERModel` loads Hugging Face's token classification pipeline with `aggregation_strategy="simple"` so WordPiece subword tokens merge back into entity spans with `entity_group`, `start`, and `end` offsets (`src/core/models/nlp.py`). `_format_ner_entities` guards against negative or inverted spans and falls back to `text[start:end]` if the pipeline omits `word`.

**Q10: How does `ZeroShotClassificationModel` handle single-label vs. multi-label scoring and abuse protection?**  
`multi_label=False` applies NLI softmax normalization across candidate hypotheses so scores sum to `1.0`; `multi_label=True` scores each hypothesis independently. Results are explicitly sorted descending by rounded probability. `NLPService.validate_zero_shot_inputs` strips/deduplicates labels and enforces `<= 25` unique labels and `<= 200` chars per label to block NLI combinatorial blowup.

**Q11: How does semantic keyword extraction rank keyphrases?**  
`extract_candidate_ngrams` (`src/core/nlp_utils.py`) extracts alphanumeric n-grams (`1 <= min_n <= max_n <= 3`), strips phrases whose first or last word is in `ENGLISH_STOPWORDS`, requires tokens >= 2 chars with at least one letter, and caps at `KEYWORD_MAX_CANDIDATES=100` (or evaluates caller-supplied `candidates`). `EmbeddingModel` tokenizes `[clean_text, *candidates]` up to `max_length=256`, applies attention-masked mean pooling, L2-normalizes vectors (`p=2`), computes cosine similarity via NumPy dot products, and breaks score ties deterministically by `(-score, keyword)`. In batch mode (`deduplicated_batched_embeddings`), identical documents and candidate strings across the entire batch are deduplicated before embedding!

**Q12: How does summarization handle documents longer than `distilbart-cnn-12-6`'s context limit?**  
`chunk_text_by_sentences()` (`src/core/nlp_utils.py`) splits along sentence (and if needed word) boundaries respecting both `SUMMARIZATION_CHUNK_SIZE=1800` characters and `max_tokens=600` (counted via the HF tokenizer when loaded or `max(words*1.35, chars/3.5)` fallback), keeping chunks safely below DistilBART's `1024`-token ceiling. `NLPService.run_summarization_model()` summarizes each chunk and consolidates chunk summaries across up to `5` rounds, breaking early when only one chunk remains or when `consolidated_input` stops shrinking (`no_progress`). Usage metering charges `cost_units = len(initial_chunks)`.

**Q13: How does Ollama extraction enforce schema compliance and avoid hallucinated placeholders?**  
`ExtractionService` (`src/services/extraction_service.py`) validates the schema recursively upfront (`_validate_schema_node` ensures root `"type": "object"` with `"properties"`), prompts `qwen2.5:3b` via `OllamaClient.generate(..., format="json")`, strips Markdown code fences, parses JSON, coerces scalars/arrays/enums, and converts `"none"`, `"n/a"`, `"unknown"`, `"not specified"`, `"not mentioned"`, `""`, and `"null"` to JSON `None` (`null`). Contextually, for `string` fields, `_is_placeholder_string` uses a word-boundary regex against `source_text` so legitimate occurrences like `"status unknown"` or `"none reported"` actually written in the source document are preserved instead of stripped! Cost units bill proportionally: `max(1, round((total_tokens * 5) / 1000))`.

### Model Lifecycle, Batching & Testing

**Q14: How does `ModelRegistry` handle concurrent requests for the same model vs. different models, LRU eviction, and CUDA OOM?**  
`get_model()` uses per-model `_loading_locks[model_id]`: parallel requests for the same unloaded model wait on that lock and initialize weights once, while distinct models load concurrently under `_registry_lock`'s `max_cached_models` reservation (`_loaded_models + _pending_loads`). When full, `_ensure_cache_capacity()` evicts the oldest `last_used_at` idle model (`not is_busy()`), raising `503 ModelResourceError` if every slot is busy. When a PyTorch OOM string occurs inside `BaseModel._run_worker`, `DeviceManager.handle_cuda_oom()` logs live vs. reserved VRAM diagnostics and calls `gc.collect()` + `torch.cuda.empty_cache()`, noting accurately that `empty_cache()` reclaims only unoccupied caching-allocator blocks rather than live model tensors.

**Q15: How do tests separate fast offline determinism from real database and model checks?**  
The default `pytest` command uses `-m "not smoke"` (offline SQLite + mocked pipeline/Ollama tests covering all routes, schemas, concurrency, and failure paths) so contributors do not need Hugging Face internet downloads, GPU hardware, or a live Ollama daemon. Real PostgreSQL 16 schema, constraint, cascade, and deadlock tests live behind `@pytest.mark.postgres` (`NLPULSE_TEST_POSTGRES_URL`), and live Hugging Face inference sits behind `@pytest.mark.smoke` (`RUN_REAL_MODEL_SMOKE=1`).

---

## 3. Verified Engineering Challenges Solved

| Challenge | Original Problem | Root Cause | Fix & Verification | Tradeoff |
| :--- | :--- | :--- | :--- | :--- |
| **1. Empty-Scope Privilege Escalation** | Passing `scopes: []` when creating an API key silently granted `analyze:read` | `APIKeyCreate` allowed empty lists and `APIKeyService.create_key` used `scopes or ["analyze:read"]` | Enforced `min_length=1` + non-blank validator in `src/schemas/api_key.py`; switched service check to `if scopes is None` and rejected empty lists; verified in `test_m8_empty_scopes_cannot_escalate_to_analyze_read` | Explicit legacy default requires callers to omit `scopes` rather than pass `[]` |
| **2. Unscoped Key & Usage Routes** | Keys holding only e.g. `sentiment:read` could hit `/api/v1/keys` or `/api/v1/usage/summary` | Router endpoints checked `get_current_developer` without `require_scopes(...)` | Added `keys:write` / `admin`, `keys:read` / `keys:write` / `admin`, and `usage:read` / `admin` checks on `/keys` and `/usage/summary`; updated bootstrap key scopes in `developers.py` and verified `test_m8_key_and_usage_endpoints_enforce_scopes` | Old custom keys created without `keys:*`/`usage:read` can no longer manage keys/usage |
| **3. PostgreSQL AB-BA Deadlock** | Concurrent batch requests timed out waiting against live PostgreSQL 16.15 | `assert_within_quota`/`reserve_quota` held PostgreSQL `FOR UPDATE` row locks across slow ML inference while `settle_batch_usage` waited on `QuotaReservationManager`'s asyncio lock | Removed `FOR UPDATE` across inference, keeping mutual exclusion inside `QuotaReservationManager` + atomic in-lock commit; verified zero deadlock & exact `10/10` quota in `test_postgres_real.py` (1.17s) | Shifts multi-request lock coordination cleanly to process memory (single-worker boundary) |
| **4. Single-Request Quota Race & Double-Counting** | Parallel single NLP requests raced past quota (`check_quota`), and naive `finally:` teardown briefly double-counted committed rows on PG 16 (`3` of `4` allowed requests passed); mocked `AsyncMock` sessions later failed on `session.info.get(...)` | Single routes only checked quota pre-inference (`check_quota`); releasing in `finally:` ran after `record_usage` committed `+1` to DB; `AsyncMock.info` returned a coroutine | Added `reserve_single_quota` + `session.info["_quota_reservations"]` settled atomically inside `record_usage`'s lock (`isinstance(session_info, dict)` guard); verified exact `4/4` pass on PG 16.15 + all 16 unit tests | Assumes single-request `UsageService(db)` receives the FastAPI request `AsyncSession` |
| **5. Registry Cache-Limit Race** | 3 parallel distinct `get_model()` calls against `max_cached_models=2` loaded 3 models | Capacity check saw only `_loaded_models`, ignoring concurrent in-flight loads | Tracked `_pending_loads` and `_evicting_models` under `_registry_lock` inside `_ensure_cache_capacity()`; verified with `test_concurrent_loading_beyond_cache_limit_respects_max_cached` | Concurrent loads exceeding cache fail fast with `503` if existing slots are busy |
| **6. Slashed HF Model ID Routing (`404`)** | `/models/dslim/bert-base-NER` and `/unload` returned `404` | Single-segment `{model_id}` failed to match Hugging Face `org/model` slashes | Used `{model_id:path}` with `/unload` ordered before `GET /models/{model_id:path}`; verified in `test_m8_model_routes_support_slashed_huggingface_ids` | Path suffix `/unload` is reserved after `/models/` |

---

## 4. Honest Answers About Current Limitations

- **Why is single-worker deployment documented as the supported boundary?**  
  `SlidingWindowRateLimiter`, `QuotaReservationManager`, and `ModelRegistry` keep per-key windows, in-flight quota reservations, and PyTorch weights in process memory. Under `--workers > 1`, workers would enforce independent rate limits (`N * limit`), lose cross-worker reservation visibility until DB commit, and duplicate loaded Transformer models (`N * memory`). Moving to multi-worker requires Redis-backed sliding windows and quota leases, plus out-of-process model serving if GPU VRAM is shared.
- **Why not claim CUDA acceleration as verified?**  
  Although the development workstation reports an `NVIDIA GeForce RTX 4050 Laptop GPU` and `DeviceManager` implements `torch.cuda` inspection/OOM handling (verified via unit mocks), `.venv` currently installs CPU-only PyTorch (`torch 2.15.0+cpu`, `cuda_available=False`), so physical CUDA kernel execution has not been tested.
- **Why was live Ollama inference not executed in CI/pytest?**  
  Running Ollama requires an external running daemon and downloaded 3B model weights; testing against mocked HTTP transports keeps the suite fast, deterministic, and portable, while error mapping (`503`, `404`, `504`, `502`) and schema grounding stay thoroughly exercised.
- **What else would broader production deployment require?**  
  End-to-end live CUDA and Ollama smoke verification inside containerized GPU staging, distributed Redis rate limiting and quota leases for horizontal scaling, observability exporters (Prometheus/OpenTelemetry beyond structured logs), TLS termination at a reverse proxy, and an admin/billing upgrade API (since `developers` public registration restricts to `tier="free"`).

---

## 5. Planned vs. Implemented Boundary (Future Improvements Only)

These remain **future engineering extensions**, not implemented features:
- Redis-backed distributed `RateLimiter` and atomic quota leases for `--workers > 1` or multi-node clusters.
- Wiring `MODEL_ENABLE_WARMUP` cleanly with readiness probes (`/ready` vs `/health`) if preloading specific models becomes necessary.
- End-to-end automated GPU CI runs validating CUDA PyTorch wheels and live Ollama daemon evaluation datasets.
- Persistent async job queues (`202 Accepted` + job polling) for much larger multi-hundred-document summarization/extraction batches.
