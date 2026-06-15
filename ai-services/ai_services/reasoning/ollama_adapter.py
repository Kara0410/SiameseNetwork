"""Reasoning provider backed by a local Ollama server.

Falls back to `TemplateReasoningProvider` if Ollama is unreachable, returns an empty
response, or errors - the verification API never fails because of LLM availability.
"""

import httpx

from .base import ReasoningProvider
from .prompting import build_prompt
from .schemas import VerificationContext
from .template_engine import TemplateReasoningProvider

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"


class OllamaReasoningProvider(ReasoningProvider):
    name = "ollama"

    def __init__(self, host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL, timeout: float = 8.0) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._fallback = TemplateReasoningProvider()

    async def explain(self, context: VerificationContext) -> str:
        prompt = build_prompt(context)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.host}/api/generate",
                    json={"model": self.model, "prompt": prompt, "stream": False},
                )
                response.raise_for_status()
                text = response.json().get("response", "").strip()
                return text or await self._fallback.explain(context)
        except Exception:
            return await self._fallback.explain(context)
