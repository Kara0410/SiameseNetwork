"""Turn raw `[0, 1]` heatmaps into frontend-friendly representations."""

import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def heatmap_to_overlay(heatmap: np.ndarray, base_image: Image.Image, alpha: float = 0.45) -> str:
    """Resize `heatmap` to `base_image`'s size, colorize it, and blend it on top.

    Returns a `data:image/png;base64,...` data URI ready to drop into an `<img src>`.
    """
    base = base_image.convert("RGB")
    width, height = base.size

    resized = cv2.resize(heatmap.astype(np.float32), (width, height))
    colored = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    base_arr = np.array(base).astype(np.float32)
    blended = base_arr * (1 - alpha) + colored.astype(np.float32) * alpha
    image = Image.fromarray(blended.astype(np.uint8))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def heatmap_to_grid(heatmap: np.ndarray, grid_size: int = 8) -> list[list[float]]:
    """Downsample `heatmap` to a `grid_size x grid_size` grid of floats in `[0, 1]`.

    Lightweight enough to send over JSON for the dashboard's CSS-grid heatmap.
    """
    resized = cv2.resize(heatmap.astype(np.float32), (grid_size, grid_size))
    resized -= resized.min()
    max_value = resized.max()
    if max_value > 0:
        resized /= max_value
    return [[round(float(v), 3) for v in row] for row in resized]
