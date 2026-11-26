"""Application-level exception hierarchy for NLPulse."""

from typing import Any, Optional


class NLPulseException(Exception):
    """Base exception for all domain errors in NLPulse."""

    def __init__(self, message: str, status_code: int = 500, details: Optional[Any] = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details


class AuthenticationError(NLPulseException):
    """Raised when authentication fails (missing, invalid, expired, or revoked API key)."""

    def __init__(self, message: str = "Invalid, missing, or revoked API key.") -> None:
        super().__init__(message=message, status_code=401)


class PermissionDeniedError(NLPulseException):
    """Raised when an API key lacks the required scopes for an endpoint."""

    def __init__(self, message: str = "Insufficient permissions to access this resource.") -> None:
        super().__init__(message=message, status_code=403)


class ResourceNotFoundError(NLPulseException):
    """Raised when a requested resource does not exist or does not belong to the tenant."""

    def __init__(self, message: str = "Requested resource not found.") -> None:
        super().__init__(message=message, status_code=404)


class ResourceConflictError(NLPulseException):
    """Raised when a unique constraint or conflicting state is violated (e.g. duplicate email)."""

    def __init__(self, message: str = "Resource conflict.") -> None:
        super().__init__(message=message, status_code=409)


class RateLimitExceededError(NLPulseException):
    """Raised when the process-local rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded. Please slow down.", retry_after: int = 60) -> None:
        super().__init__(message=message, status_code=429, details={"retry_after": retry_after})
        self.retry_after = retry_after


class QuotaExceededError(NLPulseException):
    """Raised when a developer has exhausted their monthly request quota."""

    def __init__(self, message: str = "Monthly request quota exhausted. Upgrade your tier to increase quota.") -> None:
        super().__init__(message=message, status_code=429)


class ModelError(NLPulseException):
    """Base exception for model lifecycle and inference errors."""

    def __init__(self, message: str = "A model lifecycle error occurred.", status_code: int = 500) -> None:
        super().__init__(message=message, status_code=status_code)


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not registered in the model registry."""

    def __init__(self, model_id: str) -> None:
        super().__init__(message=f"Model '{model_id}' is not registered.", status_code=404)
        self.model_id = model_id


class ModelLoadError(ModelError):
    """Raised when a model fails to initialize or load weights."""

    def __init__(self, model_id: str, reason: str) -> None:
        super().__init__(message=f"Failed to load model '{model_id}': {reason}", status_code=500)
        self.model_id = model_id
        self.reason = reason


class ModelInUseError(ModelError):
    """Raised when attempting to unload or evict a model actively executing inference."""

    def __init__(self, model_id: str, active_inferences: int = 1) -> None:
        super().__init__(
            message=f"Model '{model_id}' is currently busy with {active_inferences} active inference task(s).",
            status_code=409,
        )
        self.model_id = model_id
        self.active_inferences = active_inferences


class ModelResourceError(ModelError):
    """Raised when model cache limits or system memory limits prevent loading."""

    def __init__(self, message: str = "Insufficient resources to load model into memory.") -> None:
        super().__init__(message=message, status_code=503)


class DeviceUnavailableError(ModelError):
    """Raised when a requested hardware device (e.g. CUDA) is not available."""

    def __init__(self, requested_device: str, available_devices: list[str]) -> None:
        super().__init__(
            message=f"Requested device '{requested_device}' is unavailable. Available: {available_devices}",
            status_code=400,
        )
        self.requested_device = requested_device
        self.available_devices = available_devices


class ModelInferenceError(ModelError):
    """Raised when an error occurs during model inference."""

    def __init__(self, model_id: str, reason: str) -> None:
        super().__init__(message=f"Inference failed for model '{model_id}': {reason}", status_code=500)
        self.model_id = model_id
        self.reason = reason


class OllamaError(NLPulseException):
    """Base exception for local generative Ollama inference errors."""

    def __init__(self, message: str = "Ollama generative service error.", status_code: int = 502) -> None:
        super().__init__(message=message, status_code=status_code)


class OllamaUnavailableError(OllamaError):
    """Raised when the local Ollama daemon service cannot be reached (503 Service Unavailable)."""

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        super().__init__(
            message=f"Local Ollama service unavailable at {base_url}. Ensure the Ollama daemon is running.",
            status_code=503,
        )
        self.base_url = base_url


class OllamaModelNotFoundError(OllamaError):
    """Raised when the requested generative model is not found in local Ollama instance (404 Not Found)."""

    def __init__(self, model_name: str) -> None:
        super().__init__(
            message=f"Generative model '{model_name}' not found in local Ollama service. Run 'ollama pull {model_name}'.",
            status_code=404,
        )
        self.model_name = model_name


class OllamaInferenceError(OllamaError):
    """Raised when Ollama returns an error or generation fails (502 Bad Gateway)."""

    def __init__(self, reason: str, status_code: int = 502) -> None:
        super().__init__(message=f"Generative inference failed: {reason}", status_code=status_code)
        self.reason = reason


class OllamaTimeoutError(OllamaInferenceError):
    """Raised when an upstream Ollama request exceeds its timeout (504 Gateway Timeout)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason=reason, status_code=504)


class ExtractionValidationError(NLPulseException):
    """Raised when generated extraction payload violates the user-specified schema (422 Unprocessable Entity)."""

    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(message=message, status_code=422, details=details)

