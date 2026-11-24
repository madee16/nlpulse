"""Unit tests for cryptographic API key security."""

from src.core.security import generate_api_key, hash_api_key, verify_api_key


def test_generate_api_key():
    """Verify generated API key entropy, structure, and prefix."""
    raw_key, prefix, key_hash = generate_api_key(prefix="nlp_live")
    assert raw_key.startswith("nlp_live_")
    assert prefix.startswith("nlp_live_")
    assert len(prefix) == 16
    assert len(key_hash) == 64
    assert hash_api_key(raw_key) == key_hash


def test_verify_api_key():
    """Verify constant-time key hash verification."""
    raw_key, _, key_hash = generate_api_key()
    assert verify_api_key(raw_key, key_hash) is True
    assert verify_api_key(raw_key + "invalid", key_hash) is False
    assert verify_api_key("nlp_live_wrong", key_hash) is False
