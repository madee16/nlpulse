"""Hardware detection and device management for model execution.

Provides dynamic detection of CUDA GPUs, CPU fallback, device validation,
VRAM tracking, and CUDA out-of-memory mitigation.
"""

from __future__ import annotations

import gc
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from src.core.exceptions import DeviceUnavailableError
from src.core.logging import get_logger

logger = get_logger("nlpulse.core.device")

DeviceType = Literal["cuda", "cpu"]


@dataclass(frozen=True)
class HardwareInfo:
    """Detected hardware environment and compute capabilities."""

    device_type: str
    cuda_available: bool
    device_name: str
    device_count: int
    total_vram_mb: Optional[float]
    allocated_vram_mb: Optional[float]
    reserved_vram_mb: Optional[float]
    cpu_count: int
    system_ram_mb: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert hardware info to a JSON-serializable dictionary."""
        return {
            "device_type": self.device_type,
            "cuda_available": self.cuda_available,
            "device_name": self.device_name,
            "device_count": self.device_count,
            "total_vram_mb": self.total_vram_mb,
            "allocated_vram_mb": self.allocated_vram_mb,
            "reserved_vram_mb": self.reserved_vram_mb,
            "cpu_count": self.cpu_count,
            "system_ram_mb": self.system_ram_mb,
        }


class DeviceManager:
    """Detects and manages computational devices for model lifecycle execution."""

    def __init__(self, preference: str = "auto") -> None:
        """Initialize device manager with a preferred device strategy.

        Args:
            preference: Preferred device ("auto", "cuda", "cpu").
        """
        self._preference = preference.lower()
        self._target_device = self._detect_target_device()
        logger.info(
            "DeviceManager initialized",
            extra={
                "preference": self._preference,
                "target_device": self._target_device,
                "cuda_available": self.is_cuda_available(),
            },
        )

    @property
    def target_device(self) -> str:
        """Return the active resolved target device string."""
        return self._target_device

    @property
    def preference(self) -> str:
        """Return configured device preference."""
        return self._preference

    @staticmethod
    def is_cuda_available() -> bool:
        """Check whether CUDA is genuinely available via PyTorch."""
        try:
            import torch

            return bool(torch.cuda.is_available())
        except (ImportError, Exception):
            return False

    def _detect_target_device(self) -> str:
        """Detect and validate the target execution device based on preference and hardware."""
        cuda_ok = self.is_cuda_available()

        if self._preference == "cuda":
            if not cuda_ok:
                available = ["cpu"]
                raise DeviceUnavailableError("cuda", available)
            return "cuda"
        elif self._preference == "cpu":
            return "cpu"
        elif self._preference == "auto":
            return "cuda" if cuda_ok else "cpu"
        else:
            logger.warning(
                f"Unknown device preference '{self._preference}', falling back to 'auto'",
                extra={"preference": self._preference},
            )
            return "cuda" if cuda_ok else "cpu"

    def resolve_device(self, requested_preference: Optional[str] = None) -> str:
        """Resolve a specific device preference or default to configured target.

        Args:
            requested_preference: Optional override ("auto", "cuda", "cpu").

        Returns:
            Resolved device identifier ("cuda" or "cpu").

        Raises:
            DeviceUnavailableError: If the requested device is unavailable.
        """
        if requested_preference is None:
            return self._target_device

        pref = requested_preference.lower()
        cuda_ok = self.is_cuda_available()

        if pref == "cuda":
            if not cuda_ok:
                raise DeviceUnavailableError("cuda", ["cpu"])
            return "cuda"
        elif pref == "cpu":
            return "cpu"
        elif pref == "auto":
            return "cuda" if cuda_ok else "cpu"
        else:
            raise DeviceUnavailableError(pref, ["cuda", "cpu"] if cuda_ok else ["cpu"])

    def get_hardware_info(self) -> HardwareInfo:
        """Inspect and return current hardware capabilities and memory metrics."""
        cuda_ok = self.is_cuda_available()
        device_name = "CPU"
        device_count = 0
        total_vram_mb: Optional[float] = None
        allocated_vram_mb: Optional[float] = None
        reserved_vram_mb: Optional[float] = None

        if cuda_ok:
            try:
                import torch

                device_count = torch.cuda.device_count()
                if device_count > 0:
                    device_name = torch.cuda.get_device_name(0)
                    props = torch.cuda.get_device_properties(0)
                    total_vram_mb = round(props.total_memory / (1024 * 1024), 2)
                    allocated_vram_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
                    reserved_vram_mb = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2)
            except Exception as e:
                logger.warning(f"Error querying CUDA properties: {e}")
                cuda_ok = False

        # CPU & RAM inspection
        cpu_count = os.cpu_count() or 1
        system_ram_mb: Optional[float] = None

        try:
            if sys.platform == "win32":
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    system_ram_mb = round(stat.ullTotalPhys / (1024 * 1024), 2)
            elif hasattr(os, "sysconf"):
                pages = os.sysconf("SC_PHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                system_ram_mb = round((pages * page_size) / (1024 * 1024), 2)
        except Exception as e:
            logger.debug(f"Unable to read physical RAM metrics: {e}")

        return HardwareInfo(
            device_type=self._target_device,
            cuda_available=cuda_ok,
            device_name=device_name,
            device_count=device_count,
            total_vram_mb=total_vram_mb,
            allocated_vram_mb=allocated_vram_mb,
            reserved_vram_mb=reserved_vram_mb,
            cpu_count=cpu_count,
            system_ram_mb=system_ram_mb,
        )

    def empty_cuda_cache(self) -> None:
        """Release unoccupied cached memory held by CUDA caching allocator.

        Note: This only returns unused cached blocks to the driver. It cannot
        free live tensor memory allocated by active models.
        """
        if self.is_cuda_available():
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception as e:
                logger.debug(f"Failed to empty CUDA cache: {e}")

    def handle_cuda_oom(self) -> Dict[str, Any]:
        """Diagnostic logging and cache release following a CUDA out-of-memory event.

        Distinguishes live tensor memory from unused cached allocator memory.
        Note: empty_cache() is a memory release mechanism for unused blocks;
        it cannot recover an operation that genuinely exceeded physical VRAM.
        Active models are not invalidated or corrupted.

        Returns:
            Dictionary containing VRAM diagnostics before and after cache release.
        """
        logger.warning(
            "CUDA Out-of-Memory event encountered. Distinguishing live allocations from cached memory."
        )
        diagnostics: Dict[str, Any] = {
            "live_allocated_mb": None,
            "reserved_mb": None,
            "reclaimable_cached_mb": None,
            "cache_flushed": False,
        }

        if not self.is_cuda_available():
            return diagnostics

        try:
            import torch

            if torch.cuda.device_count() > 0:
                allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
                reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
                reclaimable = max(0.0, reserved - allocated)
                diagnostics["live_allocated_mb"] = round(allocated, 2)
                diagnostics["reserved_mb"] = round(reserved, 2)
                diagnostics["reclaimable_cached_mb"] = round(reclaimable, 2)

                logger.warning(
                    f"CUDA OOM Diagnostics: Live active tensors = {diagnostics['live_allocated_mb']} MB, "
                    f"Allocator reserved = {diagnostics['reserved_mb']} MB, "
                    f"Unused cache reclaimable = {diagnostics['reclaimable_cached_mb']} MB. "
                    "Note: torch.cuda.empty_cache() releases only unused cache, not live model tensors."
                )
        except Exception as e:
            logger.debug(f"Unable to read CUDA memory diagnostics during OOM: {e}")

        # Trigger garbage collection and release unused cache blocks
        gc.collect()
        self.empty_cuda_cache()
        diagnostics["cache_flushed"] = True
        return diagnostics


# Module-level default singleton
device_manager = DeviceManager()
