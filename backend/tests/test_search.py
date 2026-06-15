import io

from PIL import Image


def _png_bytes(color: tuple[int, int, int]) -> io.BytesIO:
    image = Image.new("RGB", (64, 64), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_search_returns_matches_and_embedding_map_after_seeding_via_verify(client):
    client.post(
        "/api/v1/verify",
        files={
            "image_a": ("a.png", _png_bytes((10, 200, 30)), "image/png"),
            "image_b": ("b.png", _png_bytes((10, 200, 30)), "image/png"),
        },
    )

    response = client.post(
        "/api/v1/search",
        files={"image": ("q.png", _png_bytes((10, 200, 30)), "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "dummy"
    assert len(data["matches"]) >= 1
    assert len(data["embedding_map"]) >= 1
    assert "id" in data["embedding_map"][0]
    assert "x" in data["embedding_map"][0]
    assert "y" in data["embedding_map"][0]


def test_search_unknown_model_returns_400(client):
    response = client.post(
        "/api/v1/search",
        files={"image": ("q.png", _png_bytes((10, 200, 30)), "image/png")},
        params={"model": "does-not-exist"},
    )

    assert response.status_code == 400
