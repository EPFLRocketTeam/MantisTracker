import cv2
from tracking.open_tracker import OpenTracker

class Tracking:

  def __init__(self, method):
    if method == "OpenTracker":
      self.tracker = OpenTracker();
    else:
      raise ValueError("This method is not implemented.")

  def track(self, video, box=None):
    self.tracker.track(video, box)

def main():
  tracker = Tracking("OpenTracker")

  video = cv2.VideoCapture("test/test2.mp4")

  tracker.track(video)


if __name__ == '__main__':
  main()