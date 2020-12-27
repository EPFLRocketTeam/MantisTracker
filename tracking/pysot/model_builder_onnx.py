
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder

from tracking.pysot.onnx_adapter import OnnxAdapter

class ModelBuilderOnnx(ModelBuilder):
  def __init__(self):
    super(ModelBuilder, self).__init__()

    path = "tracking/training/onnx/"

    # build backbone
    self.backbone = OnnxAdapter(path + "backbone.onnx")

    # build adjust layer
    if cfg.ADJUST.ADJUST:
        self.neck = OnnxAdapter(path + "neck.onnx")

    # build rpn head
    self.rpn_head = OnnxAdapter(path + "rpn_head.onnx")

    # build mask head
    if cfg.MASK.MASK:
        self.mask_head = OnnxAdapter(path + "mask_head.onnx")

        if cfg.REFINE.REFINE:
            self.refine_head = OnnxAdapter(path + "refine_head.onnx")
