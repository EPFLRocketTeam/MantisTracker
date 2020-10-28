import cv2

class OpenTracker:

  def __init__(self):
    """
    Initialize an OpenCV tracker
    """
    self.tracker = cv2.TrackerKCF_create()


  def track(self, video, box=None):
    """
    Track an object on a video given an initial box

    video: OpenCV video
    box: the rectancle representing the initial box: e.g (a, b, c, d)

    return
    """

    succ, frame = video.read()

    if not succ:
      print("Cannot read video frame.")
      return

     # if no box is provided, draw one
    if box is None:
      box = cv2.selectROI(frame, False)

    succ = self.tracker.init(frame, box)

    if not succ:
      print("Tracker initialization failed.")
      return

    # frame counter
    count = 1

    while True:
      succ, frame = video.read()

      if not succ:
        print("Cannot read video frame.")
        break

      succ, box = self.tracker.update(frame)

      if succ:
        # Tracking success
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
      else:
        # Tracking failure
        print("Tracker update failed.")
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

      cv2.imwrite(f"tracking/result/{count}.jpg", frame)
      cv2.imshow("Tracking", frame)
      count += 1
      
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break








