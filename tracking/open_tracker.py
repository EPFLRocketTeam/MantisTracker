import cv2

class OpenTracker:

  def __init__(self, method):
    """
    Initialize an OpenCV tracker
    """
    if method == "KCF":
      self.tracker = cv2.TrackerKCF_create()
    elif method == "MOSSE":
      self.tracker = cv2.TrackerMOSSE_create()
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
