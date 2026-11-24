"""Declarative base class and model registry for NLPulse."""

from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


__all__ = ["Base"]
