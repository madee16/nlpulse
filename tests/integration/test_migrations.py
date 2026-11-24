"""Integration tests for Alembic database migrations."""

import os
import tempfile
from alembic import command
from alembic.config import Config


def test_alembic_migration_upgrade_and_downgrade():
    """Verify Alembic migration upgrade to head and downgrade to base cycle."""
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
        # Restore default test environment variable if needed
        os.environ.pop("DATABASE_URL", None)
        if os.path.exists(tmp_db_path):
            try:
                os.remove(tmp_db_path)
            except OSError:
                pass
