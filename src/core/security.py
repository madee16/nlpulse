"""Cryptographic security utilities for API key generation and hashing."""

import hashlib
import hmac
import secrets


def generate_api_key(prefix: str = "nlp_live") -> tuple[str, str, str]:
    """Generate a high-entropy API key.

    Returns:
        tuple[str, str, str]: (raw_key, key_prefix, key_hash)
            - raw_key: Returned once to the developer, NEVER stored.
            - key_prefix: Short prefix for identification and logging.
            - key_hash: SHA-256 hash stored in the database.
    """
    secret = secrets.token_urlsafe(32)
    raw_key = f"{prefix}_{secret}"
    key_prefix = raw_key[:16]
    key_hash = hash_api_key(raw_key)
    return raw_key, key_prefix, key_hash


def hash_api_key(raw_key: str) -> str:
    """Compute the SHA-256 hash of a raw API key."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def verify_api_key(raw_key: str, stored_hash: str) -> bool:
    """Verify a raw API key against a stored hash using constant-time comparison."""
    computed_hash = hash_api_key(raw_key)
    return hmac.compare_digest(computed_hash, stored_hash)
