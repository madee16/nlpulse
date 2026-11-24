"""Declarative base class and model registry for NLPulse."""

from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


# Import models so Base.metadata is fully populated for Alembic migrations
from src.models.developer import Developer  # noqa: E402
from src.models.api_key import APIKey  # noqa: E402
from src.models.usage_record import UsageRecord  # noqa: E402

__all__ = ["Base", "Developer", "APIKey", "UsageRecord"]
