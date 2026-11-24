"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import List, Optional, Union

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Application
    APP_NAME: str = "NLPulse"
    APP_ENV: str = "development"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # "json" or "text"

    # Database (PostgreSQL 16)
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgrespassword"
    POSTGRES_DB: str = "nlpulse"
    DATABASE_URL: Optional[str] = None

    # Model Lifecycle Management (M3)
    MODEL_DEVICE_PREFERENCE: str = "auto"  # "auto", "cuda", "cpu"
    MODEL_MAX_CACHED: int = 3
    MODEL_DEFAULT_CONCURRENCY: int = 2
    MODEL_INFERENCE_TIMEOUT_SECONDS: float = 30.0
    MODEL_ENABLE_WARMUP: bool = False

    # CORS
    CORS_ORIGINS: Union[List[str], str] = ["*"]

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Optional[str], info) -> str:
        """Assemble async PostgreSQL connection string if not explicitly set."""
        if isinstance(v, str) and v.strip():
            return v
        data = info.data
        user = data.get("POSTGRES_USER", "postgres")
        password = data.get("POSTGRES_PASSWORD", "postgrespassword")
        server = data.get("POSTGRES_SERVER", "localhost")
        port = data.get("POSTGRES_PORT", 5432)
        db = data.get("POSTGRES_DB", "nlpulse")
        return f"postgresql+asyncpg://{user}:{password}@{server}:{port}/{db}"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse CORS origins if provided as a comma-separated string or list."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(f"Invalid CORS_ORIGINS format: {v}")


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
