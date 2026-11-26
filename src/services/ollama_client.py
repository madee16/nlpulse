"""Asynchronous HTTP client interface communicating with local Ollama service."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx

from src.core.config import get_settings
from src.core.exceptions import (
    OllamaInferenceError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from src.core.logging import get_logger

logger = get_logger("nlpulse.services.ollama_client")
settings = get_settings()


class OllamaClient:
    """Asynchronous HTTP client communicating with local Ollama daemon."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.timeout = timeout or settings.OLLAMA_TIMEOUT_SECONDS
        self._external_client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._external_client is not None:
            return self._external_client
        return httpx.AsyncClient(timeout=self.timeout)

    async def check_health(self) -> bool:
        """Verify whether the local Ollama service is reachable and responsive.

        Returns:
            bool: True if Ollama is running and responsive, False otherwise.
        """
        try:
            client = await self._get_client()
            if self._external_client is not None:
                resp = await client.get(f"{self.base_url}/api/tags", timeout=3.0)
                return resp.status_code == 200
            else:
                async with client as c:
                    resp = await c.get(f"{self.base_url}/api/tags", timeout=3.0)
                    return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, Exception) as exc:
            logger.debug(f"Ollama health check failed: {exc}")
            return False

    async def list_models(self) -> List[Dict[str, Any]]:
        """List all models currently installed in the local Ollama instance.

        Returns:
            List of dictionaries containing model metadata.

        Raises:
            OllamaUnavailableError: If the Ollama daemon cannot be reached.
        """
        try:
            client = await self._get_client()
            if self._external_client is not None:
                resp = await client.get(f"{self.base_url}/api/tags")
            else:
                async with client as c:
                    resp = await c.get(f"{self.base_url}/api/tags")
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise OllamaUnavailableError(self.base_url) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(f"Ollama list_models timed out after {self.timeout}s.") from exc
        except Exception as exc:
            raise OllamaInferenceError(f"Unexpected error communicating with Ollama: {exc}") from exc

        if resp.status_code != 200:
            snippet = resp.text[:300]
            raise OllamaInferenceError(f"Failed to list Ollama models: HTTP {resp.status_code} - {snippet}")

        data = resp.json()
        return data.get("models", [])

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        format: Optional[str] = "json",
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute text generation request against local Ollama /api/generate endpoint.

        Args:
            prompt: Text prompt input to instruct the model.
            model: Name of local model (e.g. 'qwen2.5:3b'). Defaults to settings.OLLAMA_MODEL.
            system: Optional system instruction prompt to enforce persona or rules.
            format: Output format ('json' for JSON enforcement, or None for raw text).
            options: Optional runtime inference options (e.g. temperature, num_predict).
            timeout: Optional per-request timeout override in seconds.

        Returns:
            Dictionary containing 'response', 'prompt_tokens', 'completion_tokens', 'total_tokens'.

        Raises:
            OllamaUnavailableError: When local Ollama service is not running or unreachable.
            OllamaModelNotFoundError: When the specified model is not downloaded in Ollama.
            OllamaTimeoutError: When generation times out waiting on upstream Ollama.
            OllamaInferenceError: When generation fails or returns an invalid payload.
        """
        target_model = model or settings.OLLAMA_MODEL
        req_timeout = timeout or self.timeout

        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if format:
            payload["format"] = format
        if options:
            payload["options"] = options

        logger.debug(
            f"Sending generation request to Ollama for model '{target_model}'",
            extra={"model": target_model, "base_url": self.base_url},
        )

        try:
            client = await self._get_client()
            if self._external_client is not None:
                resp = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=req_timeout,
                )
            else:
                async with client as c:
                    resp = await c.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=req_timeout,
                    )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            logger.warning(f"Connection refused to Ollama daemon at {self.base_url}")
            raise OllamaUnavailableError(self.base_url) from exc
        except httpx.TimeoutException as exc:
            logger.warning(f"Ollama generation timed out after {req_timeout}s for model '{target_model}'")
            raise OllamaTimeoutError(f"Ollama generation timed out after {req_timeout}s.") from exc
        except Exception as exc:
            logger.error(f"Unexpected error communicating with Ollama: {exc}")
            raise OllamaInferenceError(f"Unexpected error communicating with Ollama: {exc}") from exc

        # Handle status codes
        if resp.status_code == 404:
            body = resp.text[:300]
            if "model" in body.lower() and "not found" in body.lower():
                raise OllamaModelNotFoundError(target_model)
            raise OllamaInferenceError(f"Ollama endpoint not found (HTTP 404): {body}")

        if resp.status_code != 200:
            raise OllamaInferenceError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}")

        try:
            data = resp.json()
        except Exception as exc:
            raise OllamaInferenceError(f"Ollama returned non-JSON response payload: {resp.text[:300]}") from exc

        raw_response = data.get("response", "")
        prompt_tokens = int(data.get("prompt_eval_count", 0))
        completion_tokens = int(data.get("eval_count", 0))
        total_tokens = prompt_tokens + completion_tokens

        return {
            "response": raw_response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "total_duration_ns": int(data.get("total_duration", 0)),
            "model": data.get("model", target_model),
        }
