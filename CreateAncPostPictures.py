"""Interactive webcam tool for collecting anchor/positive face images.

While the webcam preview is open: press 'a' to save the current frame
as an anchor image, 'p' to save it as a positive image, and 'q' to quit.
"""

import os
import uuid

import cv2
from matplotlib import pyplot as plt

from config import ANCHOR_DIR, POSITIVE_DIR


def create_anch_pos_img() -> None:
    """Open the webcam and let the user collect anchor/positive images."""
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()

        # Crop to a 250x250 region so faces are captured at a consistent size.
        frame = frame[250:250 + 250, 500: 500 + 250, :]

        # collect anchors
        if cv2.waitKey(1) & 0XFF == ord("a"):
            img_name = os.path.join(ANCHOR_DIR, "{}.jpg".format(uuid.uuid1()))
            cv2.imwrite(img_name, frame)

        # collect positives
        if cv2.waitKey(1) & 0XFF == ord("p"):
            img_name = os.path.join(POSITIVE_DIR, "{}.jpg".format(uuid.uuid1()))
            cv2.imwrite(img_name, frame)

        # Show image
        cv2.imshow("Image Collection", frame)

        # Breaking Webcam capture
        if cv2.waitKey(1) & 0XFF == ord("q"):
            break

    # release webcam
    cap.release()
    # close the image show frame
    cv2.destroyAllWindows()
    return plt.imshow(frame)


if __name__ == "__main__":
    create_anch_pos_img()
