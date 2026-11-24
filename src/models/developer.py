"""Developer model representing tenants in NLPulse."""

from datetime import datetime
from typing import TYPE_CHECKING, List
import uuid

from sqlalchemy import Boolean, CheckConstraint, DateTime, Integer, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base

if TYPE_CHECKING:
    from src.models.api_key import APIKey
    from src.models.usage_record import UsageRecord


class Developer(Base):
    """Developer organization/tenant entity."""

    __tablename__ = "developers"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    tier: Mapped[str] = mapped_column(
        String(50),
        default="free",
        nullable=False,
    )
    monthly_request_quota: Mapped[int] = mapped_column(
        Integer,
        default=1000,
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Table constraints
    __table_args__ = (
        CheckConstraint(
            "monthly_request_quota >= 0",
            name="ck_developers_quota_nonnegative",
        ),
    )

    # Relationships
    api_keys: Mapped[List["APIKey"]] = relationship(
        "APIKey",
        back_populates="developer",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    usage_records: Mapped[List["UsageRecord"]] = relationship(
        "UsageRecord",
        back_populates="developer",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Developer id={self.id} email={self.email} tier={self.tier}>"
