import base64
import io

from PIL import Image


def _data_uri(color: tuple[int, int, int]) -> str:
    image = Image.new("RGB", (64, 64), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def test_live_websocket_streams_similarity_and_liveness(client):
    with client.websocket_connect("/api/v1/live") as websocket:
        websocket.send_json({"image": _data_uri((120, 60, 200))})
        first = websocket.receive_json()
        assert first["model"] == "dummy"
        assert first["similarity"] == 1.0
        assert 0.0 <= first["spoof_risk"] <= 1.0
        assert isinstance(first["flags"], list)

        websocket.send_json({"image": _data_uri((120, 60, 200))})
        second = websocket.receive_json()
        assert second["similarity"] > 0.99


def test_live_websocket_reports_missing_image(client):
    with client.websocket_connect("/api/v1/live") as websocket:
        websocket.send_json({})
        response = websocket.receive_json()
        assert "error" in response
