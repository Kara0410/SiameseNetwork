import numpy as np
from PIL import Image

from ai_services.embeddings import get_model, list_models
from ai_services.embeddings.registry import available_models


def test_available_models_lists_all_backends_without_importing_them():
    assert set(available_models()) == {"dummy", "clip", "vit", "efficientnet", "siamese"}


def test_list_models_returns_metadata_for_every_backend():
    infos = {info.name: info for info in list_models()}
    assert set(infos) == {"dummy", "clip", "vit", "efficientnet", "siamese"}
    assert infos["clip"].dimension == 512
    assert infos["vit"].dimension == 768
    assert infos["efficientnet"].dimension == 1280
    assert infos["siamese"].dimension == 256
    assert infos["dummy"].explainability == "heuristic"


def test_unknown_model_raises():
    try:
        get_model("does-not-exist")
    except ValueError as exc:
        assert "does-not-exist" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown model name")


def test_dummy_embed_is_normalized_and_deterministic():
    model = get_model("dummy")
    image = Image.new("RGB", (64, 64), color=(120, 50, 200))

    vector_a = model.embed(image)
    vector_b = model.embed(image)

    assert vector_a.shape == (64,)
    assert np.allclose(vector_a, vector_b)
    assert abs(np.linalg.norm(vector_a) - 1.0) < 1e-5


def test_dummy_embed_differs_for_different_images():
    # Solid-color images are scale-invariant under the dummy model's fixed linear
    # projection + L2 normalization, so use textured (non-uniform) images instead.
    model = get_model("dummy")
    rng = np.random.default_rng(0)
    image_a = Image.fromarray(rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8))
    image_b = Image.fromarray(rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8))

    vector_a = model.embed(image_a)
    vector_b = model.embed(image_b)

    assert not np.allclose(vector_a, vector_b)


def test_dummy_explain_returns_heatmap_in_unit_range():
    model = get_model("dummy")
    image = Image.new("RGB", (64, 64), color=(10, 200, 30))

    heatmap = model.explain(image)

    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6
