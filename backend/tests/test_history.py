import io

from PIL import Image


def _png_bytes(color: tuple[int, int, int]) -> io.BytesIO:
    image = Image.new("RGB", (64, 64), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_history_is_empty_initially(client):
    response = client.get("/api/v1/history")

    assert response.status_code == 200
    assert response.json() == {"items": []}


def test_history_includes_recent_verification(client):
    verify_response = client.post(
        "/api/v1/verify",
        files={
            "image_a": ("a.png", _png_bytes((120, 60, 200)), "image/png"),
            "image_b": ("b.png", _png_bytes((120, 60, 200)), "image/png"),
        },
    )
    trace_id = verify_response.json()["trace_id"]

    response = client.get("/api/v1/history")

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 1
    assert items[0]["trace_id"] == trace_id
    assert items[0]["model"] == "dummy"


def test_history_limit_is_respected(client):
    for _ in range(3):
        client.post(
            "/api/v1/verify",
            files={
                "image_a": ("a.png", _png_bytes((120, 60, 200)), "image/png"),
                "image_b": ("b.png", _png_bytes((120, 60, 200)), "image/png"),
            },
        )

    response = client.get("/api/v1/history", params={"limit": 2})

    assert response.status_code == 200
    assert len(response.json()["items"]) == 2
