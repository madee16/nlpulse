"""In-memory sliding window rate limiter for single-node deployment.

Architecture & Concurrency Notes:
--------------------------------
1. Process-local Memory:
   This rate limiter stores state in process memory. When running under multiple Uvicorn worker
   processes (e.g. `uvicorn --workers 4`), each worker maintains its own independent history.
   For true cross-process distributed rate limiting, an external data store like Redis would be
   required. In this single-node portfolio deployment, process-local limiting provides zero-latency
   protection without adding Redis infrastructure dependencies.

2. Lock Striping for High Concurrency:
   To prevent global lock contention across all API tenants, this implementation uses a striped
   lock array (64 buckets). Requests for different API keys map to different locks, maximizing
   parallelism under concurrent multi-tenant loads.

3. Memory Safety & Automatic Pruning:
   Stale timestamps are cleaned on every check. Inactive keys with no recent requests are
   automatically evicted from memory, preventing unbounded dictionary growth.
"""

import asyncio
import time
from collections import defaultdict
from typing import Dict, List, Tuple

NUM_LOCK_STRIPES = 64


class SlidingWindowRateLimiter:
    """Thread-safe sliding window rate limiter with lock striping and memory pruning."""

    def __init__(
        self,
        default_max_requests: int = 60,
        default_window_seconds: int = 60,
        max_tracked_keys: int = 10000,
    ) -> None:
        self.default_max_requests = default_max_requests
        self.default_window_seconds = default_window_seconds
        self.max_tracked_keys = max_tracked_keys
        self._history: Dict[str, List[float]] = {}
        # Striped lock pool to reduce contention across distinct keys
        self._locks = [asyncio.Lock() for _ in range(NUM_LOCK_STRIPES)]
        self._global_lock = asyncio.Lock()

    def _get_lock(self, key: str) -> asyncio.Lock:
        """Select a striped lock based on the stable hash of the key."""
        stripe_idx = abs(hash(key)) % NUM_LOCK_STRIPES
        return self._locks[stripe_idx]

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

        lock = self._get_lock(key)
        async with lock:
            timestamps = self._history.get(key, [])
            valid_timestamps = [t for t in timestamps if t > window_start]

            if len(valid_timestamps) >= limit:
                oldest_timestamp = valid_timestamps[0]
                retry_after = max(1, int(oldest_timestamp + window - now))
                self._history[key] = valid_timestamps
                return False, retry_after

            # Check capacity safety guard
            if key not in self._history and len(self._history) >= self.max_tracked_keys:
                await self.prune_stale_entries()

            # Record current request timestamp
            valid_timestamps.append(now)
            self._history[key] = valid_timestamps
            return True, 0

    async def prune_stale_entries(self, window_seconds: int | None = None) -> int:
        """Evict all keys that have zero active timestamps in the rolling window."""
        window = window_seconds or self.default_window_seconds
        cutoff = time.time() - window

        evicted_count = 0
        async with self._global_lock:
            keys_to_remove = []
            for k, timestamps in list(self._history.items()):
                valid = [t for t in timestamps if t > cutoff]
                if not valid:
                    keys_to_remove.append(k)
                else:
                    self._history[k] = valid

            for k in keys_to_remove:
                self._history.pop(k, None)
                evicted_count += 1

        return evicted_count

    async def reset(self) -> None:
        """Clear all rate limit history."""
        async with self._global_lock:
            self._history.clear()


# Global in-memory rate limiter singleton
rate_limiter = SlidingWindowRateLimiter()
