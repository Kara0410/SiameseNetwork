"""Shared interface for LLM reasoning providers."""

from abc import ABC, abstractmethod

from .schemas import VerificationContext


class ReasoningProvider(ABC):
    """Turns a `VerificationContext` into a human-readable explanation."""

    name: str

    @abstractmethod
    async def explain(self, context: VerificationContext) -> str:
        """Return a short natural-language explanation of `context`."""
