import cv2
from open_tracker import OpenTracker

def main():
  tracker = OpenTracker()

  video = cv2.VideoCapture("test/test2.mp4")

  tracker.track(video, None)


if __name__ == '__main__':
  main()