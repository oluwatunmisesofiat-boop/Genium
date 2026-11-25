"""Application configuration managed via Pydantic Settings."""
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENV: str = "development"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Vector Storage (Qdrant)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "reference_packaging_faces"

    # Object Storage
    STORAGE_BACKEND: str = "local"  # "local" or "s3"
    STORAGE_LOCAL_DIR: Path = Path("./data/images")
    S3_BUCKET_NAME: Optional[str] = "genie-product-intelligence"
    AWS_REGION: str = "eu-west-1"

    # External APIs
    GEMINI_API_KEY: Optional[str] = None
    GOOGLE_MAPS_API_KEY: Optional[str] = None

    # NVIDIA NIM Cloud (VLM Discrepancy Review, OCR, Vision)
    NVIDIA_API_KEY: Optional[str] = None
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    NVIDIA_VLM_MODEL: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"

    # Reference paths
    CONFIGS_DIR: Path = Path("./configs")
    GREENBOOK_DB_PATH: Path = Path("./data/greenbook/greenbook.db")

    @property
    def is_production(self) -> bool:
        return self.ENV.lower() == "production"


settings = Settings()
