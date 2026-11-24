"""Unit tests for NLPulse configuration."""

import os
from unittest.mock import patch

from src.core.config import Settings, get_settings


def test_default_settings():
    """Verify default settings values when no environment variables are set."""
    settings = Settings()
    assert settings.APP_NAME == "NLPulse"
    assert settings.APP_ENV == "development"
    assert settings.APP_VERSION == "0.1.0"
    assert settings.DEBUG is False
    assert settings.API_V1_PREFIX == "/api/v1"
    assert settings.LOG_LEVEL == "INFO"
    assert settings.POSTGRES_SERVER == "localhost"
    assert settings.POSTGRES_PORT == 5432
    assert settings.POSTGRES_USER == "postgres"
    assert settings.POSTGRES_DB == "nlpulse"
    assert "postgresql+asyncpg://" in str(settings.DATABASE_URL)


def test_settings_environment_override():
    """Verify settings can be overridden via environment variables."""
    with patch.dict(
        os.environ,
        {
            "APP_NAME": "NLPulse-Staging",
            "APP_ENV": "staging",
            "DEBUG": "true",
            "LOG_LEVEL": "DEBUG",
            "POSTGRES_PORT": "5433",
        },
        clear=False,
    ):
        settings = Settings()
        assert settings.APP_NAME == "NLPulse-Staging"
        assert settings.APP_ENV == "staging"
        assert settings.DEBUG is True
        assert settings.LOG_LEVEL == "DEBUG"
        assert settings.POSTGRES_PORT == 5433


def test_get_settings_caching():
    """Verify get_settings returns a cached instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
