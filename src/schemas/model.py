"""Pydantic schemas for model lifecycle inspection and hardware metrics."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelMetadataResponse(BaseModel):
    """Public metadata representation for an AI/NLP model."""

    model_id: str
    name: str
    task: str
    version: str
    description: str
    target_device: Optional[str] = None
    vram_estimate_mb: float
    ram_estimate_mb: float
    max_concurrency: int
    timeout_seconds: float


class ModelStatusResponse(BaseModel):
    """Real-time operational status of a registered or loaded model."""

    model_id: str
    name: str
    task: str
    version: str
    state: str
    device: Optional[str] = None
    loaded_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    active_inferences: int = 0
    total_inferences: int = 0
    max_concurrency: int = 2
    vram_estimate_mb: float = 0.0
    ram_estimate_mb: float = 250.0
    error_message: Optional[str] = None


class HardwareInfoResponse(BaseModel):
    """Hardware compute and memory inspection details."""

    device_type: str = Field(..., description="Active compute device (cuda or cpu)")
    cuda_available: bool = Field(..., description="Whether CUDA execution is supported on host")
    device_name: str = Field(..., description="Hardware model name")
    device_count: int = Field(..., description="Number of detected GPU accelerators")
    total_vram_mb: Optional[float] = Field(None, description="Total dedicated GPU memory in MB")
    allocated_vram_mb: Optional[float] = Field(None, description="Currently allocated VRAM in MB")
    reserved_vram_mb: Optional[float] = Field(None, description="VRAM reserved by caching allocator in MB")
    cpu_count: int = Field(..., description="Host logical CPU core count")
    system_ram_mb: Optional[float] = Field(None, description="Total host physical system RAM in MB")


class ModelUnloadResponse(BaseModel):
    """Response returned upon model unloading or cache eviction."""

    message: str
    model_id: str
    status: str
