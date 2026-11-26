"""Usage service for recording API metrics, evaluating quotas, and aggregation."""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional
import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import QuotaExceededError
from src.models.developer import Developer
from src.models.usage_record import UsageRecord
from src.schemas.usage import UsageSummaryResponse

_NUM_QUOTA_LOCK_STRIPES = 64


class QuotaReservationManager:
    """Process-local atomic reservation tracker preventing concurrent quota TOCTOU races."""

    def __init__(self) -> None:
        self._locks: List[asyncio.Lock] = [asyncio.Lock() for _ in range(_NUM_QUOTA_LOCK_STRIPES)]
        self._locks_loop: Optional[asyncio.AbstractEventLoop] = None
        self._reserved: Dict[uuid.UUID, int] = {}

    def _ensure_loop_locks(self) -> None:
        """Rebind striped locks when accessed from a different running event loop."""
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._locks_loop is not running_loop:
            self._locks = [asyncio.Lock() for _ in range(_NUM_QUOTA_LOCK_STRIPES)]
            self._locks_loop = running_loop

    def get_lock(self, developer_id: uuid.UUID) -> asyncio.Lock:
        """Select a striped lock for a given developer tenant."""
        self._ensure_loop_locks()
        return self._locks[abs(hash(developer_id)) % _NUM_QUOTA_LOCK_STRIPES]

    def get_reserved(self, developer_id: uuid.UUID) -> int:
        """Return currently reserved in-flight quota units for developer."""
        return self._reserved.get(developer_id, 0)

    def add_reserved(self, developer_id: uuid.UUID, units: int) -> None:
        """Increment reserved quota units (caller must hold developer lock)."""
        if units > 0:
            self._reserved[developer_id] = self._reserved.get(developer_id, 0) + units

    def release_sync(self, developer_id: uuid.UUID, units: int) -> None:
        """Decrement reserved quota units (caller must hold developer lock)."""
        if units <= 0:
            return
        remaining = self._reserved.get(developer_id, 0) - units
        if remaining <= 0:
            self._reserved.pop(developer_id, None)
        else:
            self._reserved[developer_id] = remaining

    async def reset(self) -> None:
        """Clear all active reservations and rebind locks for the active event loop."""
        self._reserved.clear()
        self._locks = [asyncio.Lock() for _ in range(_NUM_QUOTA_LOCK_STRIPES)]
        try:
            self._locks_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._locks_loop = None


quota_reservation_manager = QuotaReservationManager()


class UsageService:
    """Service managing API usage metering and atomic monthly quota enforcement."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def record_usage(
        self,
        developer_id: uuid.UUID,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float,
        api_key_id: Optional[uuid.UUID] = None,
        input_character_count: int = 0,
        cost_units: int = 1,
    ) -> UsageRecord:
        """Log an API invocation atomically under the developer quota lock."""
        record = UsageRecord(
            developer_id=developer_id,
            api_key_id=api_key_id,
            endpoint=endpoint,
            method=method.upper(),
            status_code=status_code,
            input_character_count=max(0, input_character_count),
            latency_ms=max(0.0, latency_ms),
            cost_units=max(0, cost_units),
        )
        async with quota_reservation_manager.get_lock(developer_id):
            self.session.add(record)
            await self.session.commit()
            await self.session.refresh(record)
            session_info = getattr(self.session, "info", None)
            reservations = session_info.get("_quota_reservations") if isinstance(session_info, dict) else None
            if isinstance(reservations, dict) and reservations.get(developer_id, 0) > 0:
                to_release = min(reservations[developer_id], max(1, record.cost_units))
                reservations[developer_id] -= to_release
                if reservations[developer_id] <= 0:
                    reservations.pop(developer_id, None)
                quota_reservation_manager.release_sync(developer_id, to_release)
        return record

    async def get_monthly_usage_count(self, developer_id: uuid.UUID) -> int:
        """Count requests invoked by developer within the current calendar month."""
        now = datetime.now(timezone.utc)
        start_of_month = datetime(now.year, now.month, 1, 0, 0, 0, tzinfo=timezone.utc)

        stmt = (
            select(func.count(UsageRecord.id))
            .where(
                UsageRecord.developer_id == developer_id,
                UsageRecord.timestamp >= start_of_month,
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def get_usage_summary(self, developer: Developer) -> UsageSummaryResponse:
        """Calculate quota consumption for the current calendar period."""
        now = datetime.now(timezone.utc)
        current_period = now.strftime("%Y-%m")
        used = await self.get_monthly_usage_count(developer.id)
        quota = developer.monthly_request_quota
        remaining = max(0, quota - used)
        percentage = round((used / quota) * 100, 2) if quota > 0 else 100.0

        return UsageSummaryResponse(
            developer_id=developer.id,
            current_period=current_period,
            requests_used=used,
            monthly_quota=quota,
            quota_remaining=remaining,
            quota_used_percentage=percentage,
            is_quota_exceeded=(used >= quota),
        )

    async def assert_within_quota(self, developer: Developer) -> None:
        """Ensure developer has not exceeded their monthly request limit.

        Serializes checks against the process-local QuotaReservationManager lock
        without holding long-lived PostgreSQL row locks across ML/LLM inference.

        Raises:
            QuotaExceededError: If the quota has been reached or exceeded.
        """
        async with quota_reservation_manager.get_lock(developer.id):
            stmt = select(Developer).where(Developer.id == developer.id)
            result = await self.session.execute(stmt)
            current_dev = result.scalar_one_or_none() or developer

            used = await self.get_monthly_usage_count(current_dev.id)
            reserved = quota_reservation_manager.get_reserved(current_dev.id)
            if used + reserved >= current_dev.monthly_request_quota:
                raise QuotaExceededError(
                    f"Monthly request quota of {current_dev.monthly_request_quota} requests exhausted. "
                    "Please upgrade your tier to continue sending requests."
                )

    async def reserve_quota(self, developer: Developer, units: int) -> None:
        """Atomically reserve quota units for an in-flight request or batch execution.

        Prevents concurrent requests from exceeding monthly_request_quota via TOCTOU races
        in a single-node deployment without holding PostgreSQL locks across inference.
        """
        if units <= 0:
            await self.assert_within_quota(developer)
            return

        async with quota_reservation_manager.get_lock(developer.id):
            stmt = select(Developer).where(Developer.id == developer.id)
            result = await self.session.execute(stmt)
            current_dev = result.scalar_one_or_none() or developer

            used = await self.get_monthly_usage_count(current_dev.id)
            reserved = quota_reservation_manager.get_reserved(current_dev.id)
            effective_used = used + reserved
            remaining = max(0, current_dev.monthly_request_quota - effective_used)

            if effective_used >= current_dev.monthly_request_quota or units > remaining:
                if units == 1:
                    raise QuotaExceededError(
                        f"Monthly request quota of {current_dev.monthly_request_quota} requests exhausted. "
                        "Please upgrade your tier to continue sending requests."
                    )
                raise QuotaExceededError(
                    f"Insufficient monthly quota remaining ({remaining} available out of "
                    f"{current_dev.monthly_request_quota}, batch requires {units}). "
                    "Please reduce batch size or upgrade your tier."
                )

            quota_reservation_manager.add_reserved(current_dev.id, units)

    async def settle_batch_usage(
        self,
        developer_id: uuid.UUID,
        reserved_units: int,
        usage_records: List[UsageRecord],
    ) -> None:
        """Persist usage records for executed items and atomically release reservation."""
        async with quota_reservation_manager.get_lock(developer_id):
            try:
                if usage_records:
                    self.session.add_all(usage_records)
                    await self.session.commit()
            finally:
                quota_reservation_manager.release_sync(developer_id, reserved_units)

    async def release_reservation(self, developer_id: uuid.UUID, units: int) -> None:
        """Release reserved quota units without recording usage (e.g. on cancellation or error)."""
        async with quota_reservation_manager.get_lock(developer_id):
            quota_reservation_manager.release_sync(developer_id, units)
