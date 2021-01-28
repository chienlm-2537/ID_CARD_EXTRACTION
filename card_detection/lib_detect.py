import numpy as np
import cv2
from collections import defaultdict
import math


def line_detection(mask):
    """
    mask: gray image
    return: a set of line in image
    """
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    dst = cv2.Canny(mask, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    # cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    # cdstP = np.copy(cdst)

    lines = cv2.HoughLines(dst, 1, np.pi / 180, 20, None, 0, 0)

    return lines


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented


def intersection(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]



def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 

    return intersections


def get_corner(mask, intersections):
    corner1 = []
    corner2 = []
    corner3 = []
    corner4 = []
    h, w = mask.shape
    print(w, h)
    a = w/2
    b = h/2
    for point in intersections:
        point = tuple(point[0])
        if (point[0]<a) & (point[1]<b):
            corner1.append(point)
        elif (point[0]>a) & (point[1]<b):
            corner2.append(point)
        elif (point[0]<a) & (point[1]>b):
            corner4.append(point)
        elif (point[0]>a) & (point[1]>b):
            # print("[INFO] line 104",point)
            corner3.append(point)
    
    top_left = np.mean(np.array(corner1), axis=0, dtype=int)
    top_right = np.mean(np.array(corner2), axis=0, dtype=int)
    bottom_right = np.mean(np.array(corner3), axis=0, dtype=int)
    bottom_left = np.mean(np.array(corner4), axis=0, dtype=int)
    return list([top_left, top_right, bottom_right, bottom_left])


def perspective_transform(image, source_points: list, dest_points: list, out_size):
    source_points = np.float32(source_points)
    dest_points = np.float32(dest_points)

    M = cv2.getPerspectiveTransform(source_points, dest_points)

    dst = cv2.warpPerspective(image, M, out_size)

    return dst

# image = cv2.imread('mask.png')
# mask = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# lines = line_detection(mask)
# if lines is not None:
#     for i in range(0, len(lines)):
#         rho = lines[i][0][0]
#         theta = lines[i][0][1]
#         a = math.cos(theta)
#         b = math.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
#         cv2.line(image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
# cv2.imshow("image", image)
# cv2.waitKey(0)