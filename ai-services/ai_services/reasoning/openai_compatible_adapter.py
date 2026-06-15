"""Reasoning provider for any OpenAI-compatible chat completions endpoint.

Works with OpenAI itself, or any self-hosted/proxied service implementing the same
`/chat/completions` contract (e.g. vLLM, LM Studio, OpenRouter). Falls back to
`TemplateReasoningProvider` on any failure.
"""

import httpx

from .base import ReasoningProvider
from .prompting import SYSTEM_PROMPT, build_prompt
from .schemas import VerificationContext
from .template_engine import TemplateReasoningProvider

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


class OpenAICompatibleReasoningProvider(ReasoningProvider):
    name = "openai_compatible"

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 15.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._fallback = TemplateReasoningProvider()

    async def explain(self, context: VerificationContext) -> str:
        prompt = build_prompt(context)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.4,
                        "max_tokens": 200,
                    },
                )
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"].strip()
                return text or await self._fallback.explain(context)
        except Exception:
            return await self._fallback.explain(context)
