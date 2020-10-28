import cv2
from detection.detection import Detection
from tracking.tracking import Tracking

def main():
  video = cv2.VideoCapture("tracking/test/test4.mp4")
  detector = Detection("Yolo")
  tracker = Tracking("OpenTracker")

  succ, frame = video.read()

  if not succ:
    print("Cannot read video frame.")
    return

  detections, image = detector.detect(frame, show=True)

  # detections = [(name, accuracy, box), ...]
  box = detections[0][2]
  tracker.track(video, box)


if __name__ == '__main__':
  main()