"""Augments the positive image set with flipped/saturation-shifted copies.

Run as a script; prompts for confirmation before writing the augmented
images alongside the originals in the positive folder.
"""

import os
import uuid

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import POSITIVE_DIR

# Number of augmented copies generated per source image.
AUGMENTATIONS_PER_IMAGE = 9


def data_aug(img: np.ndarray) -> list:
    """Return `AUGMENTATIONS_PER_IMAGE` randomly flipped/saturated copies of `img`."""
    data = []
    for i in range(AUGMENTATIONS_PER_IMAGE):
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100), np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9, upper=1,
                                                   seed=(np.random.randint(100), np.random.randint(100)))
        data.append(img)
    return data


if __name__ == "__main__":
    all_img_og_pos = os.listdir(POSITIVE_DIR)

    answer = input("Do you want to start the data augmentation? \n")

    if answer.upper() == "YES":
        # use data augmentation on anchor and positive
        for file_name in tqdm(all_img_og_pos, total=len(all_img_og_pos), desc="Doing DataAugmentation"):
            img_path = os.path.join(POSITIVE_DIR, file_name)
            img = cv2.imread(img_path)
            augmented_images = data_aug(img)

            for image in augmented_images:
                cv2.imwrite(os.path.join(POSITIVE_DIR, '{}.jpg'.format(uuid.uuid1())), image.numpy())

        print("Finish")
    else:
        print("Okay")
