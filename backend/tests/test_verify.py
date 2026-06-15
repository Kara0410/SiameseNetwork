import io

import numpy as np
from PIL import Image


def _png_bytes(color: tuple[int, int, int]) -> io.BytesIO:
    image = Image.new("RGB", (64, 64), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def _noise_png_bytes(seed: int) -> io.BytesIO:
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_verify_identical_images_returns_verified(client):
    response = client.post(
        "/api/v1/verify",
        files={
            "image_a": ("a.png", _png_bytes((120, 60, 200)), "image/png"),
            "image_b": ("b.png", _png_bytes((120, 60, 200)), "image/png"),
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "dummy"
    assert data["verdict"] == "verified"
    assert data["similarity"] > 0.99
    assert data["heatmap_a"].startswith("data:image/png;base64,")
    assert data["heatmap_b"].startswith("data:image/png;base64,")
    assert len(data["reasoning"]) > 20
    assert data["trace_id"].startswith("viq_")


def test_verify_different_images_returns_lower_similarity_and_anomaly_or_clear_signal(client):
    response = client.post(
        "/api/v1/verify",
        files={
            "image_a": ("a.png", _noise_png_bytes(0), "image/png"),
            "image_b": ("b.png", _noise_png_bytes(1), "image/png"),
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["similarity"] < 0.99
    assert data["anomalies"]  # always at least ["no spoof signal"]


def test_verify_unknown_model_returns_400(client):
    response = client.post(
        "/api/v1/verify",
        files={
            "image_a": ("a.png", _png_bytes((10, 200, 30)), "image/png"),
            "image_b": ("b.png", _png_bytes((230, 10, 250)), "image/png"),
        },
        params={"model": "does-not-exist"},
    )

    assert response.status_code == 400
