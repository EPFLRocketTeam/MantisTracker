class Tracking:

  def __init__(self, method):
    """
    Construct a tracker

    method (string):  the name of the tracking method

    ValueError if the method does not exist
    """
    if method == "KCF":
      from src.tracking.open_tracker import OpenTracker
      self.tracker = OpenTracker("KCF");
    elif method == "MOSSE":
      from src.tracking.open_tracker import OpenTracker
      self.tracker = OpenTracker("MOSSE");
    elif method == "SiamRPN":
      from src.tracking.pysot.pysot import Pysot
      self.tracker = Pysot()
    elif method == "SiamRPN-onnx":
      from src.tracking.pysot.pysot import Pysot
      self.tracker = Pysot(optimization="onnx")
    elif method == "SiamRPN-tensorrt":
      from src.tracking.pysot.pysot import Pysot
      self.tracker = Pysot(optimization="tensorrt")
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
