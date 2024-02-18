"""Creates the anchor/positive/negative data folders.

Seeds the negative folder from a local ``lfw`` (Labelled Faces in the
Wild) directory if it's empty. Run this once before collecting images
with ``CreateAncPostPictures.py``.
"""

import os

from config import ANCHOR_DIR, NEGATIVE_DIR, POSITIVE_DIR


def create_directories() -> None:
    """Create the data folders and populate `negative` from ./lfw if empty."""
    for path in (POSITIVE_DIR, NEGATIVE_DIR, ANCHOR_DIR):
        os.makedirs(path, exist_ok=True)

    # Seed the negative folder with Labelled Faces in the Wild images, if empty.
    if not os.listdir(NEGATIVE_DIR):
        for person_dir in os.listdir("lfw"):
            for file_name in os.listdir(f"lfw/{person_dir}"):
                existing_path = os.path.join("lfw", person_dir, file_name)
                new_path = os.path.join(NEGATIVE_DIR, file_name)
                os.replace(existing_path, new_path)


if __name__ == "__main__":
    create_directories()
