from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2
import os
import numpy as np
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

cfg.MODEL.WEIGHTS = os.path.join("card_detection/models/model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

def segment_mask(image):
    """
    image: a RGB image
    """
    outputs = predictor(image)
 
    x1 = int(outputs["instances"].pred_boxes.tensor[0][0])
    y1 = int(outputs["instances"].pred_boxes.tensor[0][1])
    x2 = int(outputs["instances"].pred_boxes.tensor[0][2])
    y2 = int(outputs["instances"].pred_boxes.tensor[0][3])


    # y1 = int(y1 - 0.2  * y1)
    # x1 = int(x1 - 0.2 * x1)
    # y2 += int(0.2 * y2)
    # x2 += int(0.2 * x2)
    
    mask = outputs["instances"].pred_masks.numpy()[0][y1:y2, x1:x2]
    mask = mask * 255
    id_card = image[y1:y2, x1:x2]
    mask = np.uint8(mask)
    mask = convexHullFill(mask)
    return mask, id_card


def convexHullFill(mask):
    contours, hierachy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull_ = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_.append(hull)
    hull_ = np.array(hull_)
    mask = cv2.fillPoly(mask, pts=hull_, color=(255, 255, 255))
    return mask
    # card_detection/models/model_final.pth