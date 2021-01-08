
import argparse
import cv2
import numpy as np

import os
import sys
# Path to the Pysot library
sys.path.append(os.path.abspath("tracking/library/"))

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

class Export:

  OUTPUT_PATH = "tracking/training/export/"
    
  @staticmethod
  def to_onnx(config_path, model_path):
    """
    Export a Siam* network to an ONNX model

    config_path (string): path the pysot config (.yml)
    model_path (string): path to the pysot model (.pth)
    """

    import onnx
    import torch

    # Load the PyTorch Model

    # config for the tracker
    cfg.merge_from_file(config_path)

    model = ModelBuilder()

    # load the trained model
    map_location = lambda storage, loc: storage
    model.load_state_dict(torch.load(model_path, map_location=map_location))
    model.eval()

    # Size params
    # width = 127 # random number
    z = (1, 3, cfg.TRACK.EXEMPLAR_SIZE, cfg.TRACK.EXEMPLAR_SIZE)
    x = (1, 3, cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE)
    zf = (1, 256, 6, 6)
    xf = (1, 256, 26, 26)

    # Trace the models

    # Backbone
    x = torch.rand(x, requires_grad=True, dtype=torch.float32)
    z = torch.rand(z, requires_grad=True, dtype=torch.float32)
    # trace_and_test("backbone", model.backbone, x, dynamic_axes={'input' : {2 : 'width', 3 : 'width'}})
    Export.trace_and_test("backbone", model.backbone, x)
    Export.trace_and_test("template", model.backbone, z)

    # RPN head
    x = torch.rand(zf, requires_grad=True, dtype=torch.float32)
    y = torch.rand(xf, requires_grad=True, dtype=torch.float32)
    Export.trace_and_test("rpn_head", model.rpn_head, x, y)

    # Neck
    if cfg.ADJUST.ADJUST:
      x = torch.rand(zf, requires_grad=True, dtype=torch.float32)
      Export.trace_and_test("neck", model.neck, x)

    # Mask head
    if cfg.MASK.MASK:
      x = torch.rand(zf, requires_grad=True, dtype=torch.float32)
      y = torch.rand(xf, requires_grad=True, dtype=torch.float32)
      Export.trace_and_test("mask_head", model.mask_head, x, y)

  @staticmethod
  def trace_and_test(name, model, *input, dynamic_axes={}):
    """
    Export a Pytorch model to an ONNX model
    """

    import onnx
    import onnxruntime
    import torch

    # utility function for comparing onnx runtime and model output
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    filename = Export.OUTPUT_PATH + "onnx/" + name + ".onnx"
    torch.onnx.export(model, input, filename, input_names = ['input'], dynamic_axes=dynamic_axes, do_constant_folding=True, opset_version=11)

    # compute the output of the model
    outputs = model(*input)
    if type(outputs) is not tuple: outputs = (outputs,)

    # Check models validity
    onnx_model = onnx.load(filename)
    onnx.checker.check_model(onnx_model)

    # Run and compare
    session = onnxruntime.InferenceSession(filename)

    # compute ONNX Runtime output prediction
    ort_inputs = {session.get_inputs()[i].name: to_numpy(arg) for i, arg in enumerate(input)}
    ort_outs = session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    for i, output in enumerate(outputs):
        np.testing.assert_allclose(to_numpy(output), ort_outs[i], rtol=1e-03, atol=1e-05)

    print("Exported model " + name + " produced correct results.")

  @staticmethod
  def to_tensorrt(config_path):
    """
    Export the Siam* network ONNX models to TensorRT

    config_path (string): path the pysot config (.yml)
    """

    cfg.merge_from_file(config_path)

    models = ["backbone", "template", "rpn_head"]
    if cfg.ADJUST.ADJUST: models.append("neck")
    if cfg.MASK.MASK: models.append("mask_head")

    for model in models:
      Export.parse_and_serialize(model)

  @staticmethod
  def parse_and_serialize(name):
    """
    Export a ONNX model to TensorRT
    """
    import tensorrt as trt

    # Logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    # Flag for the network
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
 
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
      with open(Export.OUTPUT_PATH + "onnx/" + name + ".onnx", 'rb') as model:
        succ = parser.parse(model.read()) 
        if not succ:
          for error in range(parser.num_errors):
            print(parser.get_error(error))
      
      with builder.create_builder_config() as config:
        config.max_workspace_size = 1 << 20
        with builder.build_engine(network, config) as engine:
          with open(Export.OUTPUT_PATH + "tensorrt/" + name + ".engine", 'wb') as f:
            f.write(engine.serialize())
    
    print("ONNX model " + name + " exported to TensorRT")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Export tool')
  subparser = parser.add_subparsers(dest='action')

  onnx_parser = subparser.add_parser('onnx', help='Export to ONNX format')

  tensorrt_parser = subparser.add_parser('tensorrt', help='Export to TensorRT. ONNX models must already exists !')

  args = parser.parse_args()

  if args.action == "onnx":
    Export.to_onnx("tracking/training/pysot/siamrpn_alex_dwxcorr/config.yaml", "tracking/training/pysot/siamrpn_alex_dwxcorr/model.pth")
  elif args.action == "tensorrt":
    Export.to_tensorrt("tracking/training/pysot/siamrpn_alex_dwxcorr/config.yaml")