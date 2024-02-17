import tensorflow as tf
import numpy as np
import cv2
import uuid
import os
from CreateImgDirectories import pos_path
from tqdm import tqdm


def data_aug(img):
    data = []
    for i in range(9):
        # img = tf.image.stateless_random_crop(img, size=(20,20,3), seed=(1,2))
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100), np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9, upper=1,
                                                   seed=(np.random.randint(100), np.random.randint(100)))
        data.append(img)
    return data


all_img_og_pos = os.listdir(os.path.join(pos_path))

answer = input("Do you want to to start the data augmentation? \n")

if answer.upper() == "YES":
    # use data augmentation on anchor and positive
    for file_name in tqdm(all_img_og_pos, total=len(all_img_og_pos), desc="Doing DataAugmentation"):
        img_path = os.path.join(pos_path, file_name)
        img = cv2.imread(img_path)
        augmented_images = data_aug(img)

        for image in augmented_images:
            cv2.imwrite(os.path.join(pos_path, '{}.jpg'.format(uuid.uuid1())), image.numpy())

    print("Finish")
else:
    print("Okay")
