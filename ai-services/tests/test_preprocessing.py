import base64
import io

from PIL import Image

from ai_services.preprocessing.image_ops import (
    detect_and_crop_face,
    load_image_from_base64,
    load_image_from_bytes,
    resize_max_dim,
)


def _sample_png_bytes(size=(50, 50), color=(10, 20, 30)) -> bytes:
    image = Image.new("RGB", size, color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_load_image_from_bytes_and_base64_round_trip():
    data = _sample_png_bytes()

    from_bytes = load_image_from_bytes(data)
    data_uri = "data:image/png;base64," + base64.b64encode(data).decode()
    from_base64 = load_image_from_base64(data_uri)

    assert from_bytes.size == from_base64.size == (50, 50)
    assert from_bytes.mode == "RGB"


def test_load_image_from_base64_without_data_uri_prefix():
    data = _sample_png_bytes()
    raw_b64 = base64.b64encode(data).decode()

    image = load_image_from_base64(raw_b64)

    assert image.size == (50, 50)


def test_resize_max_dim_caps_largest_side():
    image = Image.new("RGB", (2000, 1000))

    resized = resize_max_dim(image, max_dim=500)

    assert max(resized.size) == 500
    assert resized.size[1] == 250  # aspect ratio preserved


def test_resize_max_dim_is_noop_when_already_small():
    image = Image.new("RGB", (100, 80))

    resized = resize_max_dim(image, max_dim=500)

    assert resized.size == (100, 80)


def test_detect_and_crop_face_returns_image_even_without_face():
    image = Image.new("RGB", (100, 100), color=(5, 5, 5))

    cropped = detect_and_crop_face(image)

    assert cropped.size[0] > 0 and cropped.size[1] > 0
