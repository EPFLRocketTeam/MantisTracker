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

  def init(self, frame, box):
    """
    Initialize the tracker with a frame and the bounding box of the object

    frame (nparray): the first frame of the tracked video

    return True (the initialization is always successful)
    """
    self.tracker.init(frame, box)
    return True

  def update(self, frame):
    """
    Predict the position of the object on the next frame

    frame (nparray): a frame of the tracked video

    return whether the update was successful, the predicted box
    """
    outputs = self.tracker.track(frame)
    box = list(map(int, outputs['bbox']))
    return len(box) > 0, box
  