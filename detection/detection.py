import cv2
from detection.yolo import Yolo

class Detection:

  def __init__(self, method):
    """
    Construct a detector

    method (string): the name of the detection method

    ValueError if the method does not exist
    """
    if method == "Yolo":
      training = "detection/training/yolo/full"
      self.detector = Yolo(f"{training}/yolo-obj.cfg", "detection/training/yolo/obj.data", f"{training}/old_yolo-obj_best.weights")
    elif method == "tiny-Yolo":
      training = "detection/training/yolo/tiny"
      self.detector = Yolo(f"{training}/yolov4-tiny.cfg", "detection/training/yolo/obj.data", f"{training}/yolov4-tiny_best.weights")
    else:
      raise ValueError("This method is not implemented.") 

  def detect(self, image):
    """
    Detect a rocket on a given image

    image (nparray): an OpenCV image

    return the sorted array of detected objects, the image with a box around the detected objects
    """
    detections, image = self.detector.detect(image)
    cv2.imwrite(f"detection/result/detection.jpg", image)
    return detections, image