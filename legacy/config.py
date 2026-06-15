"""Project-wide configuration: data locations and compute device.

All scripts import paths and the torch device from here instead of
hardcoding machine-specific values, so the project can be checked out
and run on any machine.
"""

import os

import torch

# Root folder holding the anchor/positive/negative image subfolders.
# Override with the VISIONAUTHAI_DATA_DIR environment variable to point
# at a dataset stored elsewhere.
DATA_DIR = os.environ.get(
    "VISIONAUTHAI_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
)

ANCHOR_DIR = os.path.join(DATA_DIR, "anchor")
POSITIVE_DIR = os.path.join(DATA_DIR, "positive")
NEGATIVE_DIR = os.path.join(DATA_DIR, "negative")

# Use the GPU when available, otherwise fall back to CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
