import cv2
import torch
import numpy as np

import os
import sys
# Path to the Pysot library
sys.path.append(os.path.abspath("tracking/library/"))

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

class Pysot:

  def __init__(self):
    torch.set_num_threads(1)
    # config for the tracker
    cfg.merge_from_file("tracking/training/pysot/siamrpn_alex_dwxcorr/config.yaml")
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    model = ModelBuilder()
    # load the trained model
    model.load_state_dict(torch.load("tracking/training/pysot/siamrpn_alex_dwxcorr/model.pth",
                          map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # create the tracker
    self.tracker = build_tracker(model)

  def track(self, video, box):

    succ, frame = video.read()

    if not succ:
      print("Cannot read video frame.")
      return

    # if no box is provided, draw one
    if box is None:
      box = cv2.selectROI(frame, False)

    self.tracker.init(frame, box)

    # frame counter
    count = 1

    while True:
      succ, frame = video.read()

      if not succ:
        print("Cannot read video frame.")
        break

      outputs = self.tracker.track(frame)

      box = list(map(int, outputs['bbox']))

      cv2.rectangle(frame, (box[0], box[1]),
                    (box[0] + box[2], box[1] + box[3]),
                    (0, 255, 0), 3)
      
      cv2.imwrite(f"tracking/result/{count}.jpg", frame)
      cv2.imshow("Tracking", frame)
      count += 1

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break