import numpy as np
from PIL import Image

from ai_services.liveness.heuristics import assess_liveness


def test_assess_liveness_returns_bounded_risk_for_flat_image():
    image = Image.new("RGB", (128, 128), color=(128, 128, 128))

    result = assess_liveness(image)

    assert 0.0 <= result.risk <= 1.0
    assert result.flags


def test_assess_liveness_flags_sharp_textured_image_as_lower_blur_risk():
    rng = np.random.default_rng(0)
    noisy = Image.fromarray(rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8))

    result = assess_liveness(noisy)

    assert 0.0 <= result.risk <= 1.0
    assert result.sharpness > 0
