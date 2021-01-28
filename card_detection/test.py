from card_segment import *
from lib_detect import *
import cv2
import numpy as np
import time

image = cv2.imread("/home/le.minh.chien/Pictures/Webcam/2021-01-28-094137.jpg")

start = time.time()


mask, id_card = segment_mask(image)
cv2.imwrite("mask.png", mask)
lines = line_detection(mask)
segmented = segment_by_angle_kmeans(lines)
intersections = segmented_intersections(segmented)
corner = get_corner(mask, intersections)
dst_points = [[0, 0], [500, 0], [500, 300], [0, 300]]
source_points = corner
# source_points = box

dst = perspective_transform(id_card, source_points, dest_points=dst_points, out_size=(500, 300))
cv2.imwrite("dst.png", dst)
print("Time {}".format(time.time() - start))


for point in corner:
    id_card = cv2.circle(id_card, tuple(point), radius=3, color=(0, 255, 0), thickness=4)
cv2.polylines(id_card, [np.array(corner, dtype=int)], True, (0, 0, 255), thickness=2)
cv2.imshow("Mask", id_card)
cv2.imshow("Dst", dst)
cv2.waitKey(0)