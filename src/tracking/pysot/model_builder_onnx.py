
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder

from src.tracking.pysot.onnx_adapter import OnnxAdapter
from tools.export import Export

class ModelBuilderOnnx(ModelBuilder):
  """
  Override the Pysot ModelBuilder to use ONNX runtime instead of torch
  """

  MODEL_PATH = Export.OUTPUT_PATH + "onnx/"

  def __init__(self):
    super(ModelBuilder, self).__init__()

    # build backbone
    # the template method (which initialize the tracking) has different input dimensions
    self.backbone = OnnxAdapter(self.MODEL_PATH + "template.onnx")

    # build adjust layer
    if cfg.ADJUST.ADJUST:
        self.neck = OnnxAdapter(self.MODEL_PATH + "neck.onnx")

    # build rpn head
    self.rpn_head = OnnxAdapter(self.MODEL_PATH + "rpn_head.onnx")

    # build mask head
    if cfg.MASK.MASK:
        self.mask_head = OnnxAdapter(self.MODEL_PATH + "mask_head.onnx")

        if cfg.REFINE.REFINE:
            self.refine_head = OnnxAdapter(self.MODEL_PATH + "refine_head.onnx")

  def template(self, z):
    super().template(z)
    # Change the backbone because initialization and tracking have differnt input dimensions
    self.backbone = OnnxAdapter(self.MODEL_PATH + "backbone.onnx")
