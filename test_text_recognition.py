import sys
sys.path.append("text_recognition/")

from text_recognition.recognition import TextRecognition
text_recognition_model = TextRecognition(path_to_checkpoint='text_recognition/config_text_recognition/transformerocr.pth')