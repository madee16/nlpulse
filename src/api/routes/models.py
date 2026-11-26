"""API endpoints for model registry inspection, hardware telemetry, and cache management."""

from typing import List

from fastapi import APIRouter, Depends, status

from src.api.deps import (
    get_current_api_key,
    get_model_registry,
    require_scopes,
)
from src.core.device import device_manager
from src.core.exceptions import ModelNotFoundError
from src.models.api_key import APIKey
from src.schemas.model import (
    HardwareInfoResponse,
    ModelStatusResponse,
    ModelUnloadResponse,
)
from src.services.model_registry import ModelRegistry

router = APIRouter(tags=["Models & Lifecycle"])


@router.get(
    "/models",
    response_model=List[ModelStatusResponse],
    status_code=status.HTTP_200_OK,
    summary="List all registered models and their operational status",
    dependencies=[Depends(require_scopes("models:read", "analyze:read", "admin"))],
)
async def list_models(
    current_key: APIKey = Depends(get_current_api_key),
    registry: ModelRegistry = Depends(get_model_registry),
) -> List[ModelStatusResponse]:
    """Retrieve operational lifecycle metrics and availability states for all registered models."""
    models_data = registry.list_models()
    return [ModelStatusResponse(**item) for item in models_data]


@router.post(
    "/models/{model_id:path}/unload",
    response_model=ModelUnloadResponse,
    status_code=status.HTTP_200_OK,
    summary="Explicitly unload a model from memory (Admin only)",
    dependencies=[Depends(require_scopes("admin"))],
)
async def unload_model(
    model_id: str,
    force: bool = False,
    current_key: APIKey = Depends(get_current_api_key),
    registry: ModelRegistry = Depends(get_model_registry),
) -> ModelUnloadResponse:
    """Evict a loaded model from host/accelerator memory to free resources. Requires admin privileges."""
    await registry.unload_model(model_id, force=force)
    return ModelUnloadResponse(
        message=f"Model '{model_id}' was successfully unloaded.",
        model_id=model_id,
        status="unloaded",
    )


@router.get(
    "/models/{model_id:path}",
    response_model=ModelStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get status and telemetry for a specific model",
    dependencies=[Depends(require_scopes("models:read", "analyze:read", "admin"))],
)
async def get_model_status(
    model_id: str,
    current_key: APIKey = Depends(get_current_api_key),
    registry: ModelRegistry = Depends(get_model_registry),
) -> ModelStatusResponse:
    """Retrieve status, hosting device, inference counters, and error states for a single model."""
    status_dict = registry.get_model_status(model_id)
    return ModelStatusResponse(**status_dict)


@router.get(
    "/hardware",
    response_model=HardwareInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Inspect server compute hardware and accelerator memory (Admin only)",
    dependencies=[Depends(require_scopes("admin"))],
)
async def get_hardware_telemetry(
    current_key: APIKey = Depends(get_current_api_key),
) -> HardwareInfoResponse:
    """Inspect underlying compute hardware, accelerator availability, and VRAM/RAM allocation. Requires admin."""
    hw_info = device_manager.get_hardware_info()
    return HardwareInfoResponse(**hw_info.to_dict())
