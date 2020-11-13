import cv2
import time
import argparse

from detection.detection import Detection
from tracking.tracking import Tracking

class Pipeline:

  def main(self):
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

  def detect(self, image_path):
    detector = Detection("Yolo")
    image = cv2.imread(image_path)

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
  subparser = parser.add_subparsers(dest='method')

  detection = subparser.add_parser('detect')
  detection.add_argument('--image', dest='image_path', required=True, help='Detect a rocket on an image.')

  tracker = subparser.add_parser('track')
  tracker.add_argument('--video', dest='video_source', required=True, help='Track a rocket on an video feed.')

  args = parser.parse_args()

  pipeline = Pipeline()
  method = getattr(pipeline, args.method)
  method(args.image_path)