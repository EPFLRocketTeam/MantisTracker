
import torch
import cv2
import numpy as np

import onnx
import onnxruntime

import os
import sys
# Path to the Pysot library
sys.path.append(os.path.abspath("tracking/library/"))

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

#
# Load the PyTorch Model
#

# config for the tracker
cfg.merge_from_file("experiments/siamrpn_alex_dwxcorr/config.yaml")

map_location = lambda storage, loc: storage

model = ModelBuilder()

# load the trained model
model.load_state_dict(torch.load("experiments/siamrpn_alex_dwxcorr/model.pth", map_location=map_location))
model.eval()

# Size params
width = 127 # random number
xz = (1, 3, width, width)
zf = (1, 256, 6, 6)
xf = (1, 256, 26, 26)

path = "tracking/training/onnx/"

#
# Trace the models
#

def trace_and_test(name, model, *input, dynamic_axes={}):

    # utility function for comparing onnx runtime and model output
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    filename = path + name + ".onnx"
    torch.onnx.export(model, input, filename, input_names = ['input'], dynamic_axes=dynamic_axes)

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

    print("Exported model produced correct results.")

# Backbone
x = torch.rand(xz, requires_grad=True)
trace_and_test("backbone", model.backbone, x, dynamic_axes={'input' : {2 : 'width', 3 : 'width'}})

# Neck
if cfg.ADJUST.ADJUST:
    x = torch.rand(zf, requires_grad=True)
    trace_and_test("neck", model.neck, x)

# RPN head
x = torch.rand(zf, requires_grad=True)
y = torch.rand(xf, requires_grad=True)
trace_and_test("rpn_head", model.rpn_head, x, y)

# Mask head
if cfg.MASK.MASK:
    x = torch.rand(zf, requires_grad=True)
    y = torch.rand(xf, requires_grad=True)
    trace_and_test("mask_head", model.mask_head, x, y)
