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
