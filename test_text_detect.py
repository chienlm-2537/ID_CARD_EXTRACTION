# import sys
# sys.path.append("text_detection/")
# from text_detection.detector import *
# import cv2
# import time

# detector = Detector(path_config='text_detection/ssd_mobilenet_v2/pipeline.config', path_ckpt='text_detection/ssd_mobilenet_v2/ckpt/ckpt-48',
#                         path_to_labels='text_detection/scripts/label_map.pbtxt')

# image = cv2.imread("dst.png")
# start = time.time()
# image = detector.predict(image)
# # id_image = detector.get_bb_detection(image)
# a = np.array(detector.id_boxes).reshape(np.array(detector.id_boxes).shape[1:])
# cv2.imshow("id region", a)
# end = time.time()
# print('Elapsed time: ', end - start)
# cv2.imshow('results', image)
# cv2.waitKey(0)



# Test Text detection faster
import sys
sys.path.append("text_detection_faster/")
from text_detection_faster.detector import Detector
import cv2
detection_model = Detector(path_to_model='text_detection_faster/config_text_detection/model.tflite',
                           path_to_labels='text_detection_faster/config_text_detection/label_map.pbtxt',
                           nms_threshold=0.2, score_threshold=0.3)
i = cv2.imread('text_detection_faster/test.png')
result = detection_model.draw(i)
cv2.imshow('test', result)
cv2.waitKey(0)