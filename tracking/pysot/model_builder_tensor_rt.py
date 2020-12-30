
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder

from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck

from tracking.pysot.tensor_rt_adapter import TensorRtAdapter
from tools.export import Export

class ModelBuilderTensorRt(ModelBuilder):
  MODEL_PATH = Export.OUTPUT_PATH + "tensorrt/"

  def __init__(self):
    super(ModelBuilder, self).__init__()

    # build backbone
    # the template method (which initialize the tracking) has different input dimensions
    self.backbone = TensorRtAdapter(self.MODEL_PATH + "template.engine")

    # build adjust layer
    if cfg.ADJUST.ADJUST:
        self.neck = get_neck(cfg.ADJUST.TYPE,
                              **cfg.ADJUST.KWARGS)

    # build rpn head
    self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                  **cfg.RPN.KWARGS)

    # build mask head
    if cfg.MASK.MASK:
        self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                        **cfg.MASK.KWARGS)

        if cfg.REFINE.REFINE:
            self.refine_head = get_refine_head(cfg.REFINE.TYPE)
  
  
  def template(self, z):
    super().template(z)
    # Change the backbone because initialization and tracking have differnt input dimensions
    self.backbone = TensorRtAdapter(self.MODEL_PATH + "backbone.engine")
