
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder

from tracking.pysot.onnx_adapter import OnnxAdapter

class ModelBuilderOnnx(ModelBuilder):
  PATH = "tracking/training/onnx/"

  def __init__(self):
    super(ModelBuilder, self).__init__()

    # build backbone
    # the template method (which initialize the tracking) has different input dimensions
    self.backbone = OnnxAdapter(self.PATH + "template.onnx")

    # build adjust layer
    if cfg.ADJUST.ADJUST:
        self.neck = OnnxAdapter(self.PATH + "neck.onnx")

    # build rpn head
    self.rpn_head = OnnxAdapter(self.PATH + "rpn_head.onnx")

    # build mask head
    if cfg.MASK.MASK:
        self.mask_head = OnnxAdapter(self.PATH + "mask_head.onnx")

        if cfg.REFINE.REFINE:
            self.refine_head = OnnxAdapter(self.PATH + "refine_head.onnx")

  def template(self, z):
    super().template(z)
    # Change the backbone because initialization and tracking have differnt input dimensions
    self.backbone = OnnxAdapter(self.PATH + "backbone.onnx")
