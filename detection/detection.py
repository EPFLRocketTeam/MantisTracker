import cv2
from detection.yolo import Yolo

class Detection:

  def __init__(self, method):
    if method == "Yolo":
      training = "detection/training/yolo/training_1"
      self.detector = Yolo(f"{training}/yolo-obj.cfg", "detection/training/yolo/obj.data", f"{training}/yolo-obj_best.weights")
    else:
      raise ValueError("This method is not implemented.") 

  def detect(self, image, show = False):
    """
    Given an image, detect the rocket on it
    image: an OpenCV image

    return 
    """
    detections, image = self.detector.detect(image)
    if show:
      cv2.imshow('Detection', image)
      while True:
        if cv2.waitKey() & 0xFF == ord('q'):
          break
    return detections, image

def main():
  detector = Detection("Yolo")

  image = cv2.imread("training/yolo/data/obj_test_data/ROCKET1.jpg")
  detections, image = detector.detect(image, show=True)


if __name__ == '__main__':
  main()