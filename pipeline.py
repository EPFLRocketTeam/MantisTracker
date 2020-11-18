import cv2
import time
import argparse
import glob
import sys

from detection.detection import Detection
from tracking.tracking import Tracking

class Pipeline:

  @staticmethod
  def window():
      cv2.namedWindow('MantisTracker', cv2.WINDOW_NORMAL)
      cv2.resizeWindow('MantisTracker', 1000, 1000)

  @staticmethod
  def track(video_source, tracking_method, detection_method, benchmark=False):
    """
    Track a rocket on the provided video source

    video_source

    return
    """
    video = cv2.VideoCapture(video_source)
    detector = Detection(detection_method)
    tracker = Tracking(tracking_method)

    # The window used to display the result
    if not benchmark: Pipeline.window()
      
    succ, frame = video.read()
    if not succ:
      sys.exit("Cannot read the video source !")

    # Initial detection of the rocket
    detections, image = detector.detect(frame)

    if not detections:
      print("The detection failed. Please select the object manually.")
      box = cv2.selectROI("MantisTracker", frame, False)
      if box == (0, 0, 0, 0):
        sys.exit("You cancelled the selection !")
    else:
      # detections = [(name, accuracy, box), ...]
      box = detections[0][2]

    succ = tracker.init(frame, box)

    if not succ:
      sys.exit("Tracker initialization failed.")

    # frame counter
    count = 0
     # measure the time elapsed
    t0 = time.time()

    while True:
      succ, frame = video.read()

      if not succ:
        print("Video ended.")
        break

      succ, box = tracker.update(frame)

      if succ:
        # Tracking success
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
      else:
        # Tracking failure
        print("Tracker update failed.")
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
      
      count += 1
      
      if not benchmark:
        cv2.imwrite(f"tracking/result/{count}.jpg", frame)
        cv2.imshow("MantisTracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    
    t1 = time.time() - t0
    if benchmark:
      print("Time elapsed: {:.2f}s".format(t1))
      print("Performance: {:.2f} fps".format(count/t1))    

  @staticmethod
  def detect(images_path, method, benchmark=False):
    """
    Detect a rocket on the provided images.

    images_path (list(string)): the path to the images
    method (string): the method of detection
    benchmark (boolean): if set, the method will measure the performance of the detection. The images are not shown. 

    return
    """
    detector = Detection(method)

    # The window used to display the result
    if not benchmark: window()

    # open the images
    images = map(lambda path: cv2.imread(path), images_path)
    
    # measure the time elapsed
    t0 = time.time()
    # number of detections
    detected = 0

    for image in images:

      if image is None:
        sys.exit("The image could not be found !")

      detections, image = detector.detect(image)

      if detections:
        detected += 1

      if not benchmark:
        cv2.imshow('MantisTracker', image)
        while True:
          if cv2.waitKey() & 0xFF == ord('q'):
            break

    t1 = time.time() - t0
    if benchmark:
      print("Time elapsed: {:.2f}s".format(t1))
      print("{:d} detections for {:d} images, {:.2f}% accuracy.".format(detected, len(images_path), detected/len(images_path)*100))
      print("Performance: {:.2f} fps".format(len(images_path)/t1))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run the Mantis Tracker')
  subparser = parser.add_subparsers(dest='action')

  parser.add_argument('--benchmark', dest='benchmark', action='store_const', const=True, default=False,
                      help='Measure the performance of the process in frame per second')

  detection = subparser.add_parser('detect', help='Detect a rocket on an image')
  detection.add_argument('--images', dest='images_path', nargs='+', required=True, 
                      help='The path to the images to be detected')
  detection.add_argument('--method', dest='method', choices=['Yolo', 'tiny-Yolo'], default='Yolo',
                      help='The detection method to use')

  tracker = subparser.add_parser('track', help='Track a rocket on an video feed.')
  tracker.add_argument('--video', dest='video_source', required=True, help='The path to the video source')
  tracker.add_argument('--method', dest='method', choices=['OpenTracker', 'SiamRPN'], default='OpenTracker',
                      help='The tracking method to use')
  tracker.add_argument('--detection', dest='detection', choices=['Yolo', 'tiny-Yolo'], default='Yolo',
                      help='The detection method to use')
  
  args = parser.parse_args()

  if args.action == "track":
    Pipeline.track(args.video_source, args.method, args.detection, args.benchmark)
  elif args.action == "detect":
    Pipeline.detect(args.images_path, args.method, args.benchmark)

  cv2.destroyAllWindows()