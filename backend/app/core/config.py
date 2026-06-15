"""Application settings, sourced from environment variables (or `.env`).

All settings are prefixed `VISIONIQ_` so the backend, frontend, and Docker compose
files can share a single `.env` without name collisions.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "VisionIQ API"

    # Default embedding backend (one of: dummy | clip | vit | efficientnet | siamese).
    embedding_default: str = "dummy"

    # Reasoning provider (one of: template | ollama | openai_compatible).
    reasoning_provider: str = "template"

    vector_store_dir: str = "data/vector_store"
    database_url: str = "sqlite:///./data/visioniq.db"

    cors_origins: list[str] = ["http://localhost:3000"]

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(env_prefix="VISIONIQ_", env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
