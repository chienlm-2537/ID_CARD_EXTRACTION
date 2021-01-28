import sys
sys.path.append("text_detection/")


from card_detection.card_segment import *
from card_detection.lib_detect import *
from text_detection.detector import Detector
from completedModel import CompletedModel



import numpy as np
import time
import cv2

model = CompletedModel()

cap = cv2.VideoCapture(0)
print("Camera ready ", cap.isOpened())
while cap.isOpened():
    _, image = cap.read()
    if not _:
        break
    model.predict(image)
    if model.cropped_image is not None:
        cv2.imshow("ID CARD", model.cropped_image)
    cv2.imshow("image", image)
    k = cv2.waitKey(5)
    if k == 32:
        break


# # Khoi tao text detector
# detector = Detector(path_config='text_detection/ssd_mobilenet_v2/pipeline.config', path_ckpt='text_detection/ssd_mobilenet_v2/ckpt/ckpt-48',
#                         path_to_labels='text_detection/scripts/label_map.pbtxt')


# image = cv2.imread("/home/le.minh.chien/Pictures/Webcam/2021-01-22-140136.jpg")
# start = time.time()

# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     _, image = cap.read()
#     if not _:
#         break
#     mask, id_card = segment_mask(image)
#     cv2.imshow("mask", mask)
#     lines = line_detection(mask)
#     segmented = segment_by_angle_kmeans(lines)
#     intersections = segmented_intersections(segmented)
#     corner = get_corner(mask, intersections)

#     dst_points = [[0, 0], [500, 0], [500, 300], [0, 300]]
#     source_points = corner
#     # print(source_points)
#     dst = perspective_transform(id_card, source_points, dest_points=dst_points, out_size=(500, 300))
#     cv2.imwrite("dst.png", dst)
#     dst = detector.predict(dst)
#     print("Time {}".format(time.time() - start))


#     for point in corner:
#         id_card = cv2.circle(id_card, tuple(point), radius=3, color=(0, 255, 0), thickness=4)

#     cv2.polylines(id_card, [np.array(corner, dtype=int)], True, (0, 0, 255), thickness=2)
#     cv2.imshow("Mask", id_card
#     )
#     cv2.imshow("Dst", dst)   

#     k = cv2.waitKey(5)
#     if k == 32:
#         break


# mask, id_card = segment_mask(image)

# lines = line_detection(mask)
# segmented = segment_by_angle_kmeans(lines)
# intersections = segmented_intersections(segmented)
# corner = get_corner(mask, intersections)

# dst_points = [[0, 0], [500, 0], [500, 300], [0, 300]]
# source_points = corner

# dst = perspective_transform(id_card, source_points, dest_points=dst_points, out_size=(500, 300))
# cv2.imwrite("dst.png", dst)
# dst = detector.predict(dst)
# print("Time {}".format(time.time() - start))


# for point in corner:
#     id_card = cv2.circle(id_card, tuple(point), radius=3, color=(0, 255, 0), thickness=4)

# cv2.polylines(id_card, [np.array(corner, dtype=int)], True, (0, 0, 255), thickness=2)
# cv2.imshow("Mask", mask)
# cv2.imshow("Dst", dst)
# cv2.waitKey(0)