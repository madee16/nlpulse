"""Integration tests for Alembic database migrations."""

import os
import tempfile
from alembic import command
from alembic.config import Config


def test_alembic_migration_upgrade_and_downgrade():
    """Verify Alembic migration upgrade to head and downgrade to base cycle."""
    has_prev_url = "DATABASE_URL" in os.environ
    prev_db_url = os.environ.get("DATABASE_URL")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        tmp_db_path = tmp_file.name

    try:
        db_url = f"sqlite+aiosqlite:///{tmp_db_path}"
        os.environ["DATABASE_URL"] = db_url

        ini_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "alembic.ini"))
        alembic_cfg = Config(ini_path)
        alembic_cfg.set_main_option("script_location", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "migrations")))

        # 1. Upgrade to head
        command.upgrade(alembic_cfg, "head")

        # 2. Downgrade to base
        command.downgrade(alembic_cfg, "base")

        # 3. Re-upgrade to head to ensure clean repeatability
        command.upgrade(alembic_cfg, "head")
    finally:
        if has_prev_url and prev_db_url is not None:
            os.environ["DATABASE_URL"] = prev_db_url
        else:
            os.environ.pop("DATABASE_URL", None)
        if os.path.exists(tmp_db_path):
            try:
                os.remove(tmp_db_path)
            except OSError:
                pass


def test_alembic_migration_preserves_existing_database_url():
    """Verify migration tests restore pre-existing DATABASE_URL instead of wiping it."""
    sentinel = "postgresql+asyncpg://sentinel:pw@localhost:5432/original_db"
    had_prev = "DATABASE_URL" in os.environ
    prev_val = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = sentinel
    try:
        test_alembic_migration_upgrade_and_downgrade()
        assert os.environ.get("DATABASE_URL") == sentinel
    finally:
        if had_prev and prev_val is not None:
            os.environ["DATABASE_URL"] = prev_val
        else:
            os.environ.pop("DATABASE_URL", None)


def test_postgres_migration_safety_guard_rejects_arbitrary_databases():
    """Verify destructive PostgreSQL migration check blocks non-test database names."""
    import pytest
    from tests.integration.test_postgres_real import validate_safe_postgres_migration_url

    assert validate_safe_postgres_migration_url("postgresql+asyncpg://u:p@localhost:5432/nlpulse_m9_test") == "nlpulse_m9_test"
    for unsafe_url in (
        "sqlite+aiosqlite:///:memory:",
        "postgresql+asyncpg://u:p@localhost:5432/postgres",
        "postgresql+asyncpg://u:p@localhost:5432/nlpulse",
        "postgresql+asyncpg://u:p@localhost:5432/production",
    ):
        with pytest.raises(ValueError):
            validate_safe_postgres_migration_url(unsafe_url)
