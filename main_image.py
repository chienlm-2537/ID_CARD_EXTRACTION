import sys
sys.path.append("text_detection/")


from card_detection.card_segment import *
from card_detection.lib_detect import *
from text_detection.detector import Detector
from completedModel import CompletedModel



import numpy as np
import time
import cv2
import argparse


def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image_path', help='link to image', default='/home/le.minh.chien/Downloads/Datasets/OCR_CMT_DATA_ANNOTATED-20210119T054159Z-001/OCR_CMT_DATA_ANNOTATED/corner/Corner/24131045_1531592600222060_522786088386565461_n.jpg')
    return arg.parse_args()


arguments = get_args()
image_path = arguments.image_path

image = cv2.imread(image_path)

model = CompletedModel()

start = time.time()
model.predict(image)
print("Time {}".format(time.time() - start))
if model.cropped_image is not None:
    cv2.imshow("ID CARD", model.cropped_image)
else:
    assert "Can't detect ID Card in Image"
cv2.imshow("image", image)
cv2.waitKey(0)