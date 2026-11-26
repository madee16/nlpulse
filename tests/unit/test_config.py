"""Unit tests for NLPulse configuration and runtime packaging alignment."""

import ast
import os
from pathlib import Path
import re
import sys
import tomllib
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


def test_batch_settings_defaults_and_overrides():
    """Verify M7 batch processing configuration defaults and env overrides."""
    default_settings = Settings()
    assert default_settings.BATCH_MAX_SIZE == 20
    assert default_settings.BATCH_MAX_CONCURRENCY == 4
    assert default_settings.BATCH_INFERENCE_CHUNK_SIZE == 8
    assert default_settings.BATCH_EXTRACTION_CONCURRENCY == 2

    with patch.dict(
        os.environ,
        {
            "BATCH_MAX_SIZE": "50",
            "BATCH_MAX_CONCURRENCY": "6",
            "BATCH_INFERENCE_CHUNK_SIZE": "16",
            "BATCH_EXTRACTION_CONCURRENCY": "3",
        },
        clear=False,
    ):
        overridden = Settings()
        assert overridden.BATCH_MAX_SIZE == 50
        assert overridden.BATCH_MAX_CONCURRENCY == 6
        assert overridden.BATCH_INFERENCE_CHUNK_SIZE == 16
        assert overridden.BATCH_EXTRACTION_CONCURRENCY == 3


def test_pyproject_runtime_dependencies_cover_src_imports():
    """Verify runtime third-party imports in src/ are declared in pyproject.toml and requirements.txt."""
    repo_root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))

    def _pkg_name(spec: str) -> str:
        return re.split(r"[\[\s<>=!~;#]", spec.strip(), maxsplit=1)[0].lower().replace("_", "-")

    runtime_deps = {_pkg_name(dep) for dep in pyproject["project"]["dependencies"]}
    req_lines = [
        line.strip()
        for line in (repo_root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    req_deps = {_pkg_name(line) for line in req_lines}

    # Explicit regression check: httpx is imported at runtime by OllamaClient
    assert "httpx" in runtime_deps
    assert "httpx" in req_deps

    import_to_dist = {
        "pydantic_settings": "pydantic-settings",
        "starlette": "fastapi",
    }
    discovered_third_party: set[str] = set()
    for py_file in (repo_root / "src").rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".", 1)[0]
                    if top not in sys.stdlib_module_names and top != "src":
                        discovered_third_party.add(import_to_dist.get(top, top.replace("_", "-").lower()))
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                top = node.module.split(".", 1)[0]
                if top not in sys.stdlib_module_names and top != "src":
                    discovered_third_party.add(import_to_dist.get(top, top.replace("_", "-").lower()))

    missing_in_pyproject = discovered_third_party - runtime_deps
    missing_in_requirements = discovered_third_party - req_deps
    assert not missing_in_pyproject, f"Missing runtime deps in pyproject.toml: {missing_in_pyproject}"
    assert not missing_in_requirements, f"Missing runtime deps in requirements.txt: {missing_in_requirements}"
