"""Triplet dataset for fine-tuning `SiameseEmbeddingNet`.

Modernized from `legacy/SN-TripletLoss/DatasetTriplet.py`: type-hinted, configurable
image size/transform, and driven by explicit directory arguments instead of a
module-level `config.py` import.
"""

import random
from itertools import product
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class TripletFaceDataset(Dataset):
    """Random `(anchor, positive, negative)` triplets drawn from three image folders."""

    def __init__(
        self,
        anchor_dir: str,
        positive_dir: str,
        negative_dir: str,
        transform: transforms.Compose | None = None,
        num_triplets: int | None = None,
    ) -> None:
        self.transform = transform or DEFAULT_TRANSFORM

        anchors = _list_images(anchor_dir)
        positives = _list_images(positive_dir)
        negatives = _list_images(negative_dir)
        if not anchors or not positives or not negatives:
            raise ValueError(
                "anchor/positive/negative directories must each contain at least one image: "
                f"got {len(anchors)}/{len(positives)}/{len(negatives)}"
            )

        sample_size = num_triplets or (len(positives) + len(negatives))
        all_triplets = list(product(anchors, positives, negatives))
        self.triplets = random.choices(all_triplets, k=sample_size)

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int):
        anchor_path, positive_path, negative_path = self.triplets[index]
        return (
            self.transform(Image.open(anchor_path).convert("RGB")),
            self.transform(Image.open(positive_path).convert("RGB")),
            self.transform(Image.open(negative_path).convert("RGB")),
        )


def _list_images(directory: str) -> list[str]:
    path = Path(directory)
    if not path.exists():
        return []
    return [str(p) for p in sorted(path.iterdir()) if p.is_file()]
