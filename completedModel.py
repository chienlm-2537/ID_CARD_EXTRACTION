from card_detection.lib_detect import line_detection
import sys
sys.path.append("card_detection/")
sys.path.append("text_detection_faster/")
from card_detection.card_segment import *
from text_detection_faster.detector import *
from card_detection.lib_detect import *
from text_detection_faster.utils.image_utils import sort_text
from text_recognition.recognition import TextRecognition
import time

class CompletedModel():
    def __init__(self):
        self.text_detection = Detector(path_to_model='text_detection_faster/config_text_detection/model.tflite',
                                             path_to_labels='text_detection_faster/config_text_detection/label_map.pbtxt',
                                             nms_threshold=0.2, score_threshold=0.2)
        self.text_recognition_model = TextRecognition(path_to_checkpoint='text_recognition/config_text_recognition/transformerocr.pth')
        
        self.name_boxes = None
        self.birth_boxes = None
        self.add_boxes = None
        self.home_boxes = None
        self.cropped_image = None
    
    def segment_id(self, image):
        mask, id_card = segment_mask(image)
        lines = line_detection(mask)
        segmented = segment_by_angle_kmeans(lines)
        intersections = segmented_intersections(segmented)
        corner = get_corner(mask, intersections)

        dst_points = [[0, 0], [500, 0], [500, 300], [0, 300]]
        source_points = corner
        dst = perspective_transform(id_card, source_points, dest_points=dst_points, out_size=(500, 300))

        return dst
    
    def detect_text(self, image):

        detection_boxes, detection_classes, category_index = self.text_detection.predict(image)
        self.text_detection.draw(self.cropped_image)
        self.id_boxes, self.name_boxes, self.birth_boxes, self.home_boxes, self.add_boxes = sort_text(detection_boxes, detection_classes)

    
    def text_recognition(self, image):
        field_dict = dict()

        def crop_and_recog(boxes):
            crop = []
            if len(boxes) == 1:
                ymin, xmin, ymax, xmax = boxes[0]
                crop.append(image[ymin:ymax, xmin:xmax])
            else:
                for box in boxes:
                    ymin, xmin, ymax, xmax = box
                    # cv2.imwrite('./crop/test_' + str(ymin) + '_' + str(ymax) + '.png', image[ymin:ymax, xmin:xmax])
                    crop.append(image[ymin:ymax, xmin:xmax])

            return crop

        list_ans = list(crop_and_recog(self.id_boxes))
        list_ans.extend(crop_and_recog(self.name_boxes))
        list_ans.extend(crop_and_recog(self.birth_boxes))
        list_ans.extend(crop_and_recog(self.add_boxes))
        list_ans.extend(crop_and_recog(self.home_boxes))

        start1 = time.time()
        result = self.text_recognition_model.predict_on_batch(np.array(list_ans))
        end1 = time.time()
        print("predicted time: ", end1 - start1)
        field_dict['id'] = result[0]
        field_dict['name'] = ' '.join(result[1:len(self.name_boxes) + 1])
        field_dict['birth'] = result[len(self.name_boxes) + 1]
        field_dict['home'] = ' '.join(result[len(self.name_boxes) + 2: -len(self.home_boxes)])
        field_dict['add'] = ' '.join(result[-len(self.home_boxes):])
        print(field_dict)

    def predict(self, image):
        self.cropped_image = self.segment_id(image)
        self.detect_text(self.cropped_image)
        self.text_recognition(self.cropped_image)     

          
# image = cv2.imread("/home/le.minh.chien/Pictures/Webcam/2021-01-28-094137.jpg")
# model = CompletedModel()
# model.predict(image)

