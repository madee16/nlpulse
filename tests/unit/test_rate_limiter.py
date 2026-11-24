"""Unit tests for the sliding window rate limiter."""

import asyncio
import pytest

from src.core.rate_limiter import SlidingWindowRateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_allows_under_limit():
    """Verify rate limiter allows requests under the maximum limit."""
    limiter = SlidingWindowRateLimiter(default_max_requests=5, default_window_seconds=10)
    for _ in range(5):
        allowed, retry_after = await limiter.check("test_key")
        assert allowed is True
        assert retry_after == 0


@pytest.mark.asyncio
async def test_rate_limiter_blocks_over_limit():
    """Verify rate limiter rejects requests when limit is exceeded."""
    limiter = SlidingWindowRateLimiter(default_max_requests=3, default_window_seconds=5)
    for _ in range(3):
        allowed, _ = await limiter.check("test_key_limit")
        assert allowed is True

    # 4th request must be rejected
    allowed, retry_after = await limiter.check("test_key_limit")
    assert allowed is False
    assert retry_after > 0


@pytest.mark.asyncio
async def test_rate_limiter_resets_after_window():
    """Verify rate limiter allows requests after the window expires."""
    limiter = SlidingWindowRateLimiter(default_max_requests=2, default_window_seconds=1)
    allowed1, _ = await limiter.check("quick_key")
    allowed2, _ = await limiter.check("quick_key")
    assert allowed1 is True and allowed2 is True

    blocked, _ = await limiter.check("quick_key")
    assert blocked is False

    # Wait for 1-second window to roll over
    await asyncio.sleep(1.05)

    allowed_after, _ = await limiter.check("quick_key")
    assert allowed_after is True


@pytest.mark.asyncio
async def test_rate_limiter_independent_keys():
    """Verify rate limiter tracks different keys independently."""
    limiter = SlidingWindowRateLimiter(default_max_requests=1, default_window_seconds=10)
    allowed_a, _ = await limiter.check("key_a")
    allowed_b, _ = await limiter.check("key_b")
    assert allowed_a is True
    assert allowed_b is True

    blocked_a, _ = await limiter.check("key_a")
    blocked_b, _ = await limiter.check("key_b")
    assert blocked_a is False
    assert blocked_b is False
