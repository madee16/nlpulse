"""Usage service for recording API metrics, evaluating quotas, and aggregation."""

from datetime import datetime, timezone
from typing import Optional
import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import QuotaExceededError
from src.models.developer import Developer
from src.models.usage_record import UsageRecord
from src.schemas.usage import UsageSummaryResponse


class UsageService:
    """Service managing API usage metering and monthly quota enforcement."""

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
        """Log an API invocation."""
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
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
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

        Raises:
            QuotaExceededError: If the quota has been reached or exceeded.
        """
        used = await self.get_monthly_usage_count(developer.id)
        if used >= developer.monthly_request_quota:
            raise QuotaExceededError(
                f"Monthly request quota of {developer.monthly_request_quota} requests exhausted. "
                "Please upgrade your tier to continue sending requests."
            )
