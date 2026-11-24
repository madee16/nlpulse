"""Schemas for usage records and quota summaries."""

from datetime import datetime
from typing import Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field


class UsageRecordResponse(BaseModel):
    """Detailed record of an API invocation."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    endpoint: str
    method: str
    status_code: int
    input_character_count: int
    latency_ms: float
    cost_units: int
    timestamp: datetime


class UsageSummaryResponse(BaseModel):
    """Monthly usage aggregation and quota status for a developer."""

    developer_id: uuid.UUID
    current_period: str = Field(..., description="Calendar period in YYYY-MM format")
    requests_used: int = Field(..., description="Total requests invoked in the current billing period")
    monthly_quota: int = Field(..., description="Total allowed requests per month for developer tier")
    quota_remaining: int = Field(..., description="Remaining requests before 429 quota exhaustion")
    quota_used_percentage: float = Field(..., description="Percentage of monthly quota consumed")
    is_quota_exceeded: bool = Field(..., description="True if requests_used >= monthly_quota")
