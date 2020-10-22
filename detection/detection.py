import cv2
from yolo import Yolo

def main():
  training = "training/yolo/training_1"
  detector = Yolo(f"{training}/yolo-obj.cfg", "training/yolo/obj.data", f"{training}/yolo-obj_best.weights")

  image = cv2.imread("training/yolo/data/obj_test_data/ROCKET1.jpg")
  result = detector.detect(image)

  while True:
    cv2.imshow('Detection', cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    if cv2.waitKey() & 0xFF == ord('q'):
      break


if __name__ == '__main__':
  main()