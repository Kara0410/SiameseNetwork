"""Shared configuration for the ai_services package.

All values are overridable via environment variables so the package behaves the
same whether it is imported by the FastAPI backend, a training script, or a test
suite.
"""

import os

import torch


def get_device() -> torch.device:
    """Pick the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()

# Where HuggingFace/torchvision pretrained weights are cached. Override to point at a
# shared volume (e.g. in Docker) so weights survive container restarts.
MODEL_CACHE_DIR = os.environ.get(
    "VISIONIQ_MODEL_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "visioniq", "models")
)

# Optional path to a fine-tuned Siamese checkpoint produced by
# `python -m ai_services.siamese.train`. When unset, the Siamese backend uses an
# ImageNet-pretrained ResNet18 trunk with a randomly-initialized projection head.
SIAMESE_CHECKPOINT = os.environ.get("VISIONIQ_SIAMESE_CHECKPOINT")
