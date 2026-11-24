"""UsageRecord model for tracking API invocations, quotas, and audit history."""

from datetime import datetime
from typing import TYPE_CHECKING, Optional
import uuid

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base

if TYPE_CHECKING:
    from src.models.api_key import APIKey
    from src.models.developer import Developer


class UsageRecord(Base):
    """Usage record logging an individual API request.

    Note on cost_units:
        `cost_units` represents an abstract compute resource weight (e.g. 1 unit for
        standard sentiment inference, 5 units for heavy summarization), NOT a monetary currency.
    """

    __tablename__ = "usage_records"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    developer_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("developers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    api_key_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    endpoint: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    method: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
    )
    status_code: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    input_character_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    latency_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    cost_units: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    # Check constraints and composite indexes
    __table_args__ = (
        CheckConstraint(
            "input_character_count >= 0",
            name="ck_usage_records_char_count_nonnegative",
        ),
        CheckConstraint(
            "latency_ms >= 0",
            name="ck_usage_records_latency_nonnegative",
        ),
        CheckConstraint(
            "status_code >= 100 AND status_code <= 599",
            name="ck_usage_records_status_code_valid",
        ),
        CheckConstraint(
            "cost_units >= 0",
            name="ck_usage_records_cost_units_nonnegative",
        ),
        Index("ix_usage_records_developer_id_timestamp", "developer_id", "timestamp"),
    )

    # Relationships
    developer: Mapped["Developer"] = relationship(
        "Developer",
        back_populates="usage_records",
    )
    api_key: Mapped[Optional["APIKey"]] = relationship(
        "APIKey",
        back_populates="usage_records",
    )

    def __repr__(self) -> str:
        return f"<UsageRecord id={self.id} endpoint={self.endpoint} status={self.status_code}>"
