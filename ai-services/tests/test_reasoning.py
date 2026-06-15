import pytest

from ai_services.reasoning.engine import build_reasoning_provider
from ai_services.reasoning.schemas import VerificationContext


def _context(**overrides) -> VerificationContext:
    defaults = dict(
        model_name="clip",
        similarity=0.95,
        distance=0.18,
        verdict="verified",
        spoof_risk=0.05,
        anomalies=["lighting variance"],
    )
    defaults.update(overrides)
    return VerificationContext(**defaults)


@pytest.mark.asyncio
async def test_template_provider_mentions_model_and_verdict():
    provider = build_reasoning_provider("template")

    text = await provider.explain(_context())

    assert "CLIP" in text
    assert len(text) > 20


@pytest.mark.asyncio
async def test_template_provider_handles_blocked_verdict_and_high_spoof_risk():
    provider = build_reasoning_provider("template")
    context = _context(
        model_name="vit",
        similarity=0.4,
        distance=0.7,
        verdict="blocked",
        spoof_risk=0.8,
        anomalies=["replay signature"],
    )

    text = await provider.explain(context)

    assert "VIT" in text
    assert "replay signature" in text
    assert "spoof risk" in text.lower()


@pytest.mark.asyncio
async def test_ollama_provider_falls_back_to_template_when_unreachable():
    provider = build_reasoning_provider("ollama", host="http://localhost:1", model="x", timeout=0.5)

    text = await provider.explain(_context())

    assert "CLIP" in text
    assert len(text) > 20


@pytest.mark.asyncio
async def test_openai_compatible_provider_falls_back_to_template_when_unreachable():
    provider = build_reasoning_provider(
        "openai_compatible",
        api_key="invalid",
        base_url="http://localhost:1",
        model="x",
        timeout=0.5,
    )

    text = await provider.explain(_context())

    assert "CLIP" in text


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        build_reasoning_provider("not-a-provider")
