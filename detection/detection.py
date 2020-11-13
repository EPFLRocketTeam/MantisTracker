import cv2
from detection.yolo import Yolo

class Detection:

  def __init__(self, method):
    if method == "Yolo":
      training = "detection/training/yolo/result"
      self.detector = Yolo(f"{training}/yolo-obj.cfg", "detection/training/yolo/obj.data", f"{training}/yolo-obj_best.weights")
    else:
      raise ValueError("This method is not implemented.") 

  def detect(self, image):
    """
    Given an image, detect the rocket on it
    image: an OpenCV image

    return 
    """
    detections, image = self.detector.detect(image)
    cv2.imwrite(f"detection/result/detection.jpg", image)
    return detections, image