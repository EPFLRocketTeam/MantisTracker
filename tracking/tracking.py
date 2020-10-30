import cv2
from tracking.open_tracker import OpenTracker
from tracking.pysot import Pysot

class Tracking:

  def __init__(self, method):
    if method == "OpenTracker":
      self.tracker = OpenTracker();
    elif method == "SiamRPN":
      self.tracker = Pysot()
    else:
      raise ValueError("This method is not implemented.")

  def track(self, video, box=None):
    self.tracker.track(video, box)
