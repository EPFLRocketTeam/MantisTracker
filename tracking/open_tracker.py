import cv2

class OpenTracker:

  def __init__(self):
    """
    Initialize an OpenCV tracker
    """
    self.tracker = cv2.TrackerKCF_create()

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
