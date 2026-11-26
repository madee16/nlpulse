# ADR-005: Batch Processing and Bounded Concurrency

- **Status:** Accepted (Current)

## Context

Supporting bulk analysis across all six NLP capabilities requires maximizing throughput where transformer pipelines benefit from vectorized tensor batching, while preventing memory exhaustion, uncontrolled task spawning, single bad items failing entire batches, and runaway concurrency against heavy seq2seq summarization or Ollama inference on workstation hardware (`6 GB VRAM / 16 GB RAM`).

## Decision

Implement `BatchService` (`src/services/batch_service.py`) over six `/api/v1/nlp/batch/*` endpoints using capability-tailored strategies and strict resource bounds (`BATCH_MAX_SIZE=20`, `BATCH_MAX_CONCURRENCY=4`, `BATCH_INFERENCE_CHUNK_SIZE=8`, `BATCH_EXTRACTION_CONCURRENCY=2`):
- **True Micro-Batched Tensor Passes:** `sentiment` and `ner` (`batched_inference`), plus label-grouped `classification` (`grouped_batched_inference`), batch up to 8 texts per forward pass (`predict_batch`), falling back per item if a chunk fails so healthy siblings survive.
- **Cross-Document Deduplicated Embeddings:** `keywords` (`deduplicated_batched_embeddings`) extracts per-item n-gram candidates, deduplicates identical document/candidate texts across the batch, encodes unique strings in bounded chunks, and scores cosine similarities in NumPy.
- **Bounded Async Worker Pools:** `summarization` (`bounded_concurrent`, `<= BATCH_MAX_CONCURRENCY`) and `extraction` (`bounded_concurrent`, `min(BATCH_MAX_CONCURRENCY, BATCH_EXTRACTION_CONCURRENCY)`) pull from an `asyncio.Queue` in a fixed coroutine pool (`_run_bounded_workers`), cancelling cleanly on `asyncio.CancelledError`.
- **Two-Stage Validation:** Invalid batch envelopes (`[]` or `> BATCH_MAX_SIZE`) reject with `422`; invalid individual items map to `BatchItemError` inside `results[i]` while preserving order and IDs.

## Alternatives Considered

1. **Unbounded `asyncio.gather()` across every submitted item:** Spikes memory and hits `ModelInferenceError` semaphore queue timeouts (`max_concurrency=2`) under larger payloads.
2. **Naive per-item serial loops for all capabilities:** Loses vectorized PyTorch speedups on Sentiment, NER, Zero-Shot, and redundant candidate embeddings in Keyword Extraction.

## Consequences

- **Positive:** Predictable bounded resource utilization, exact position/ID preservation, partial-failure resilience, and zero cross-contamination from malformed batch items.
- **Negative:** Worst-case fallback inside a poisoned micro-batch executes failed chunks item-by-item, and large heterogeneous zero-shot batches with many distinct label lists fall back to smaller groups.

## Current Status

Implemented and verified via `tests/unit/test_batch_schemas.py`, `tests/unit/test_batch_models.py`, `tests/unit/test_batch_service.py`, `tests/integration/test_batch_api.py`, and real PostgreSQL 16 concurrency checks in `tests/integration/test_postgres_real.py`.
