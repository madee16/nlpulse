"""In-memory sliding window rate limiter for single-node deployment."""

import asyncio
import time
from collections import defaultdict
from typing import Dict, List, Tuple


class SlidingWindowRateLimiter:
    """Thread-safe, process-local sliding window rate limiter.

    Tracks request timestamps in-memory per identifier (e.g. API key or developer ID).
    Automatically prunes stale timestamps older than the active window.
    """

    def __init__(self, default_max_requests: int = 60, default_window_seconds: int = 60) -> None:
        self.default_max_requests = default_max_requests
        self.default_window_seconds = default_window_seconds
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def check(
        self,
        key: str,
        max_requests: int | None = None,
        window_seconds: int | None = None,
    ) -> Tuple[bool, int]:
        """Check if request is within rate limits.

        Args:
            key: Unique identifier (e.g. API key hash or developer UUID string).
            max_requests: Maximum allowed requests within the window.
            window_seconds: Duration of the rolling window in seconds.

        Returns:
            Tuple[bool, int]: (is_allowed, retry_after_seconds)
                - is_allowed: True if request is allowed, False if rate limited.
                - retry_after_seconds: Suggested wait time before retry.
        """
        limit = max_requests or self.default_max_requests
        window = window_seconds or self.default_window_seconds
        now = time.time()
        window_start = now - window

        async with self._lock:
            # Clean expired timestamps
            timestamps = self._history[key]
            valid_timestamps = [t for t in timestamps if t > window_start]

            if len(valid_timestamps) >= limit:
                # Earliest timestamp in the window determines when next request becomes available
                oldest_timestamp = valid_timestamps[0]
                retry_after = max(1, int(oldest_timestamp + window - now))
                self._history[key] = valid_timestamps
                return False, retry_after

            # Record current request timestamp
            valid_timestamps.append(now)
            self._history[key] = valid_timestamps
            return True, 0

    async def reset(self) -> None:
        """Clear all rate limit history (useful for test resets)."""
        async with self._lock:
            self._history.clear()


# Global in-memory rate limiter singleton
rate_limiter = SlidingWindowRateLimiter()
