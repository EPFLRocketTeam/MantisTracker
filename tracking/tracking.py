import cv2
from tracking.open_tracker import OpenTracker
from tracking.pysot.pysot import Pysot

class Tracking:

  def __init__(self, method):
    """
    Construct a tracker

    method (string):  the name of the tracking method
    """
    if method == "OpenTracker":
      self.tracker = OpenTracker();
    elif method == "SiamRPN":
      self.tracker = Pysot()
    elif method == "SiamRPN-onnx":
      self.tracker = Pysot(onnx=True)
    else:
      raise ValueError("This method is not implemented.")

  def init(self, frame, box):
    """
    Initialize the tracker with a frame and the bounding box of the object

    frame (nparray): the first frame of the tracked video

    return whether the initialization was successful
    """
    return self.tracker.init(frame, box)

  def update(self, frame):
    """
    Predict the position of the object on the next frame

    frame (nparray): a frame of the tracked video

    return whether the update was successful, the predicted box
    """
    return self.tracker.update(frame)
