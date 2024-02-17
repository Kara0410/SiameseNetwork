import os
import cv2
import uuid # Import uuid to generate unique image names
from matplotlib import pyplot as plt
from CreateImgDirectories import anc_path, pos_path


def create_anch_pos_img():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()

        # cut frame to 250 x 250
        frame = frame[250:250 + 250, 500: 500 + 250, :]

        # collect anchors
        if cv2.waitKey(1) & 0XFF == ord("a"):
            # creates unique filepath / filename
            img_name = os.path.join(anc_path, "{}.jpg".format(uuid.uuid1()))
            # write out anchor image
            cv2.imwrite(img_name, frame)

        # collect positives
        if cv2.waitKey(1) & 0XFF == ord("p"):
            # creates unique filepath / filename
            img_name = os.path.join(pos_path, "{}.jpg".format(uuid.uuid1()))
            # write out positive image
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

create_anch_pos_img()