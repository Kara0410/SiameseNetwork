"""Image decoding, resizing, and face-cropping utilities shared by the API and
WebSocket live-inference pipelines."""

import base64
import io

import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_DIMENSION = 1024
FACE_CROP_PADDING = 0.35

_face_cascade: cv2.CascadeClassifier | None = None


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Decode raw image bytes, apply EXIF orientation, and convert to RGB."""
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def load_image_from_base64(data_uri: str) -> Image.Image:
    """Decode a base64 string or `data:image/...;base64,...` URI into an RGB image."""
    payload = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
    return load_image_from_bytes(base64.b64decode(payload))


def resize_max_dim(image: Image.Image, max_dim: int = MAX_DIMENSION) -> Image.Image:
    """Downscale `image` so its longest side is at most `max_dim` pixels."""
    width, height = image.size
    if max(width, height) <= max_dim:
        return image
    scale = max_dim / max(width, height)
    return image.resize((round(width * scale), round(height * scale)))


def detect_and_crop_face(image: Image.Image, padding: float = FACE_CROP_PADDING) -> Image.Image:
    """Crop to the largest detected face plus `padding`, or return `image` unchanged
    if no face is found."""
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return image

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    pad_w, pad_h = int(w * padding), int(h * padding)
    left, top = max(x - pad_w, 0), max(y - pad_h, 0)
    right, bottom = min(x + w + pad_w, image.width), min(y + h + pad_h, image.height)
    return image.crop((left, top, right, bottom))
