# ADR-004: Local-First Optional Ollama Integration

- **Status:** Accepted (Current)

## Context

NLPulse exposes structured JSON information extraction powered by a generative instruction-tuned LLM (`POST /api/v1/nlp/extraction` and `POST /api/v1/nlp/batch/extraction`), while all core/advanced Transformer pipelines (`sentiment`, `ner`, `classification`, `keywords`, `summarization`), authentication, and standard tests must operate cleanly offline without requiring an active GPU LLM server or external cloud API keys.

## Decision

Decouple generative LLM execution behind an asynchronous HTTP client (`OllamaClient` in `src/services/ollama_client.py` using `httpx.AsyncClient`) targeting a local optional Ollama daemon (`OLLAMA_BASE_URL=http://localhost:11434`, default model `qwen2.5:3b`):
- Enforce JSON-mode output (`format="json"`, `stream=False`), pre-inference and post-sanitization **JSON Schema Draft 2020-12** validation (`jsonschema.Draft202012Validator` + built-in `FormatChecker`) for `schema_definition`, nullable field compilation for simplified `fields`, Markdown fence stripping, strict placeholder normalization (`_is_placeholder_string`), and factual grounding prompts in `ExtractionService` (`src/services/extraction_service.py`).
- Classify Ollama transport/status failures distinctly into `503 OllamaUnavailableError`, `404 OllamaModelNotFoundError`, `504 OllamaTimeoutError`, `502 OllamaInferenceError`, and `422 ExtractionValidationError` without logging or exposing raw LLM output or source PII in logs or error details.
- Omit Ollama from mandatory startup checks and default pytest execution; verify via mocked `httpx` transports.

## Alternatives Considered

1. **Embedded Python LLM bindings (`llama-cpp-python` / in-process Hugging Face CausalLM):** Avoids running a separate daemon, but introduces heavy C++/CUDA compilation issues on Windows, bloats base dependencies, and competes directly with the 3-model Transformer `ModelRegistry` for 6 GB VRAM.
2. **External cloud LLM APIs (OpenAI/Anthropic/Gemini):** Simpler structured outputs, but requires paid remote credentials, leaks tenant data off-host, and breaks offline reproducibility.

## Consequences

- **Positive:** Clean local-first privacy, zero startup blockers when Ollama is stopped, full testability via injected `httpx.AsyncClient` transports, and isolated VRAM lifecycle managed by the external Ollama daemon.
- **Negative:** Running live extraction locally requires installing Ollama and pulling a compatible model tag (`ollama pull qwen2.5:3b`); live daemon throughput and extraction quality remain unverified in automated tests.

## Current Status

Implemented in `src/services/ollama_client.py` and `src/services/extraction_service.py`. Verified extensively via mocked client/transport integration and unit suites (`test_ollama_client.py`, `test_extraction_service.py`, `test_extraction_api.py`).
