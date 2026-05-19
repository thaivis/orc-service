from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    api_key: str
    port: int = 8000
    max_image_size_mb: int = 10
    max_image_dimension: int = 2000
    log_level: str = "info"

    minio_endpoint: str | None = None
    minio_access_key: str | None = None
    minio_secret_key: str | None = None
    minio_bucket: str = "thaivis-id-documents"
    minio_use_ssl: bool = False

    encryption_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
