"""Integration tests for Model Lifecycle and Hardware API endpoints."""

from typing import Any
import pytest
from httpx import AsyncClient

from src.core.models.base import BaseModel, ModelMetadata
from src.main import app, lifespan
from src.services.model_registry import ModelRegistry


class DummyApiModel(BaseModel):
    """Mock model for API endpoint verification."""

    def _load_weights(self, device: str) -> None:
        self.weights = "loaded"

    def _unload_weights(self) -> None:
        self.weights = None

    def _execute_inference(self, inputs: Any, **kwargs: Any) -> Any:
        return {"output": "api-test"}


@pytest.fixture
async def registered_test_model():
    """Ensure a dummy model is registered in app registry for testing."""
    registry: ModelRegistry = app.state.model_registry if hasattr(app.state, "model_registry") else None
    if not registry:
        registry = ModelRegistry()
        app.state.model_registry = registry

    meta = ModelMetadata(
        model_id="api-dummy-model",
        name="API Dummy Model",
        task="sentiment",
        version="1.0.0",
        description="Model registered for API integration testing",
    )
    if not registry.is_registered("api-dummy-model"):
        registry.register(meta, lambda: DummyApiModel(meta))

    return "api-dummy-model"


@pytest.fixture
async def auth_keys(async_client: AsyncClient):
    """Create test developer and return standard user key and admin key."""
    # Register developer -> receives standard key
    reg_res = await async_client.post(
        "/api/v1/developers",
        json={"email": "model_dev@example.com", "name": "Model Dev"},
    )
    assert reg_res.status_code == 201
    standard_key = reg_res.json()["api_key"]

    return {
        "standard": standard_key,
    }


@pytest.mark.asyncio
async def test_list_models_unauthenticated(async_client: AsyncClient):
    """Verify GET /api/v1/models requires API key authentication."""
    response = await async_client.get("/api/v1/models")
    assert response.status_code == 401
    assert "api key required" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_list_models_authenticated(
    async_client: AsyncClient,
    auth_keys: dict,
    registered_test_model: str,
):
    """Verify GET /api/v1/models returns registered model list with valid key."""
    headers = {"X-API-Key": auth_keys["standard"]}
    response = await async_client.get("/api/v1/models", headers=headers)
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    model_ids = [m["model_id"] for m in data]
    assert registered_test_model in model_ids


@pytest.mark.asyncio
async def test_get_model_status_success(
    async_client: AsyncClient,
    auth_keys: dict,
    registered_test_model: str,
):
    """Verify GET /api/v1/models/{id} returns telemetry for an existing model."""
    headers = {"X-API-Key": auth_keys["standard"]}
    response = await async_client.get(
        f"/api/v1/models/{registered_test_model}",
        headers=headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == registered_test_model
    assert data["state"] in ["unloaded", "ready"]
    assert data["name"] == "API Dummy Model"


@pytest.mark.asyncio
async def test_get_model_status_not_found(async_client: AsyncClient, auth_keys: dict):
    """Verify GET /api/v1/models/{id} returns 404 for non-existent model."""
    headers = {"X-API-Key": auth_keys["standard"]}
    response = await async_client.get("/api/v1/models/non-existent-model", headers=headers)
    assert response.status_code == 404
    assert "not registered" in response.json()["detail"]


@pytest.mark.asyncio
async def test_unload_model_admin_scope_enforcement(
    async_client: AsyncClient,
    auth_keys: dict,
    registered_test_model: str,
):
    """Verify POST /api/v1/models/{id}/unload requires admin scope."""
    # Standard key lacks admin scope -> 403 Forbidden
    headers = {"X-API-Key": auth_keys["standard"]}
    response = await async_client.post(
        f"/api/v1/models/{registered_test_model}/unload",
        headers=headers,
    )
    assert response.status_code == 403
    assert "lacks required scope" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_hardware_telemetry_admin_scope_enforcement(
    async_client: AsyncClient,
    auth_keys: dict,
):
    """Verify GET /api/v1/hardware requires admin scope."""
    # Standard key lacks admin scope -> 403 Forbidden
    headers = {"X-API-Key": auth_keys["standard"]}
    response = await async_client.get("/api/v1/hardware", headers=headers)
    assert response.status_code == 403
    assert "lacks required scope" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_fastapi_lifespan_integration():
    """Verify application lifespan creates DeviceManager and ModelRegistry in app state."""
    async with lifespan(app):
        assert hasattr(app.state, "device_manager")
        assert app.state.device_manager is not None
        assert hasattr(app.state, "model_registry")
        assert app.state.model_registry is not None
        assert app.state.device_manager.target_device in ["cuda", "cpu"]
