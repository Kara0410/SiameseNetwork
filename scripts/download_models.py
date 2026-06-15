"""Pre-download pretrained weights for the heavier embedding backends.

Running a backend for the first time (e.g. `clip`, `vit`, `efficientnet`, `siamese`)
lazily downloads model weights from HuggingFace/torchvision on the first request,
which makes that first request very slow. Run this script ahead of time - e.g. during
a Docker image build or before a demo - so the weights are already cached under
`ai_services.config.MODEL_CACHE_DIR`.

Usage:
    python scripts/download_models.py              # all non-dummy backends
    python scripts/download_models.py clip vit     # only the named backends
"""

import sys

import numpy as np
from PIL import Image

from ai_services.embeddings.registry import available_models, get_model

_PROBE_IMAGE = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))


def main(names: list[str]) -> None:
    for name in names:
        if name == "dummy":
            continue
        print(f"Downloading weights for '{name}'...")
        model = get_model(name)
        model.embed(_PROBE_IMAGE)
        print(f"  done ({model.info.dimension}d, {model.info.explainability} explainability)")


if __name__ == "__main__":
    requested = sys.argv[1:] or [name for name in available_models() if name != "dummy"]
    main(requested)
