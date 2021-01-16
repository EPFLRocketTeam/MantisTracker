import cv2
import yaml
from src.detection.yolo import Yolo

class Detection:

  def __init__(self, method):
    """
    Construct a detector

    method (string): the name of the detection method

    ValueError if the method does not exist
    """
    # read the config file
    file = open("config.yaml")
    config = yaml.load(file, Loader=yaml.FullLoader)  

    if method == "Yolo":
      weights_path = config['yolo']['full']['weights']
      self.detector = Yolo("training/detection/yolo/full/yolo.cfg", "training/detection/yolo/obj.data", weights_path)
    elif method == "tiny-Yolo":
      weights_path = config['yolo']['tiny']['weights']
      self.detector = Yolo("training/detection/yolo/tiny/tiny_yolo.cfg", "training/detection/yolo/obj.data", weights_path)
    else:
      raise ValueError("This method is not implemented.") 

  def detect(self, image):
    """
    Detect a rocket on a given image

    image (nparray): an OpenCV image

    return the sorted array of detected objects, the image with a box around the detected objects
    """
    detections, image = self.detector.detect(image)
    return detections, image