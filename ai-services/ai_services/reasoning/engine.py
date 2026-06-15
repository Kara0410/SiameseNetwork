"""Factory for selecting a reasoning provider by name."""

from .base import ReasoningProvider
from .template_engine import TemplateReasoningProvider

PROVIDERS = ("template", "ollama", "openai_compatible")


def build_reasoning_provider(provider: str = "template", **kwargs) -> ReasoningProvider:
    """Construct a `ReasoningProvider`.

    Args:
        provider: one of `PROVIDERS`.
        **kwargs: forwarded to the provider's constructor (e.g. `host`/`model` for
            `ollama`, `api_key`/`base_url`/`model` for `openai_compatible`).
    """
    if provider == "template":
        return TemplateReasoningProvider()
    if provider == "ollama":
        from .ollama_adapter import OllamaReasoningProvider

        return OllamaReasoningProvider(**kwargs)
    if provider == "openai_compatible":
        from .openai_compatible_adapter import OpenAICompatibleReasoningProvider

        return OpenAICompatibleReasoningProvider(**kwargs)
    raise ValueError(f"Unknown reasoning provider '{provider}'. Available: {PROVIDERS}")
