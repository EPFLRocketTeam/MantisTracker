import cv2
import time
import argparse

from detection.detection import Detection
from tracking.tracking import Tracking

def main():
  video = cv2.VideoCapture("tracking/training/dataset/test/test2.mp4")
  detector = Detection("Yolo")
  tracker = Tracking("SiamRPN")

  succ, frame = video.read()

  if not succ:
    print("Cannot read video frame.")
    return

  detections, image = detector.detect(frame)

  # detections = [(name, accuracy, box), ...]
  box = detections[0][2]
  tracker.track(video, box)

def detect():
  detector = Detection("Yolo")
  image = cv2.imread("detection/training/dataset/test/ROCKET1.jpg")

  t0 = time.time()
  detections, image = detector.detect(image)
  t1 = time.time() - t0
  print("Time elapsed: ", t1)

  cv2.imshow('Detection', image)
  while True:
    if cv2.waitKey() & 0xFF == ord('q'):
      break



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run the Mantis Tracker')
  parser.add_argument('--detection', dest='method', action='store_const', const=detect, default=main,
                      help='Detect a rocket on an image.')

  args = parser.parse_args()

  args.method()