"""Heuristic anti-spoofing / liveness checks.

This is a lightweight baseline suitable for a demo: it flags obviously blurred
(replay/print) frames via Laplacian sharpness, and screen-replay/moire patterns via
the ratio of high-frequency energy in the image's 2D FFT. Neither check requires a
trained presentation-attack-detection (PAD) model, so it runs instantly on CPU.

Upgrade path: swap `assess_liveness` for a trained PAD model (e.g. a small CNN
classifier over `live` vs `spoof` crops) behind the same `LivenessResult` interface -
the FastAPI live-verification endpoint and frontend are agnostic to the
implementation.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

# Frames sharper than this (Laplacian variance) are considered in-focus.
SHARPNESS_THRESHOLD = 15.0

# Fraction of FFT energy outside the low-frequency center above which we suspect a
# screen-replay / moire pattern.
HIGH_FREQUENCY_THRESHOLD = 0.985


@dataclass(frozen=True)
class LivenessResult:
    """Result of a heuristic liveness/anti-spoofing check."""

    risk: float  # 0 (live) .. 1 (likely spoof)
    flags: list[str] = field(default_factory=list)
    sharpness: float = 0.0
    high_frequency_ratio: float = 0.0


def assess_liveness(image: Image.Image) -> LivenessResult:
    """Run heuristic liveness checks on a single frame."""
    gray = np.array(image.convert("L"), dtype=np.uint8)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    high_frequency_ratio = _high_frequency_ratio(gray)

    flags: list[str] = []
    risk = 0.0

    if sharpness < SHARPNESS_THRESHOLD:
        flags.append("low sharpness / possible blur")
        risk += 0.35

    if high_frequency_ratio > HIGH_FREQUENCY_THRESHOLD:
        flags.append("high-frequency texture pattern detected")
        risk += 0.35

    if not flags:
        flags.append("no spoof signal")

    return LivenessResult(
        risk=min(risk, 1.0),
        flags=flags,
        sharpness=sharpness,
        high_frequency_ratio=high_frequency_ratio,
    )


def _high_frequency_ratio(gray: np.ndarray) -> float:
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.abs(spectrum)

    height, width = magnitude.shape
    center_h, center_w = height // 8, width // 8
    center = magnitude[
        height // 2 - center_h : height // 2 + center_h,
        width // 2 - center_w : width // 2 + center_w,
    ]

    total_energy = magnitude.sum()
    if total_energy == 0:
        return 0.0
    return float((total_energy - center.sum()) / total_energy)
