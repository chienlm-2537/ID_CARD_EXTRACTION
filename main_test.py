import cv2
import numpy as np

mask = cv2.imread("/home/le.minh.chien/Desktop/Project/ML_IDCard_Segmentation/mask.jpeg")
mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
h, w = mask.shape[:2]
# colours, counts = np.unique(mask.reshape(-1,3), axis=0, return_counts=1)

# # Iterate through unique colours
# for index, colour in enumerate(colours):
#     count = counts[index]
#     proportion = (100 * count) / (h * w)
#     print(f"   Colour: {colour}, count: {count}, proportion: {proportion:.2f}%")

a = cv2.countNonZero(mask)
print(a)
total_pixel = h*w
print(a/total_pixel)
cv2.imshow("mask", mask)
cv2.waitKey(0)