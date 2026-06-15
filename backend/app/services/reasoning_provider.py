"""Builds the configured reasoning provider, cached for the process lifetime."""

from functools import lru_cache

from ai_services.reasoning import ReasoningProvider, build_reasoning_provider

from ..core.config import get_settings


@lru_cache
def get_reasoning_provider() -> ReasoningProvider:
    settings = get_settings()

    if settings.reasoning_provider == "ollama":
        return build_reasoning_provider(
            "ollama", host=settings.ollama_host, model=settings.ollama_model
        )
    if settings.reasoning_provider == "openai_compatible":
        return build_reasoning_provider(
            "openai_compatible",
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
        )
    return build_reasoning_provider("template")
