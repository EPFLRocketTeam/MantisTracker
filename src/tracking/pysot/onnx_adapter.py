import onnxruntime
import torch

class OnnxAdapter:

  def __init__(self, filename):
    """
    Start an ONNX inference session

    filename (string): path to the ONNX model
    """

    self.session = onnxruntime.InferenceSession(filename)

  def __call__(self, *input):
    """
    Infer using ONNX runtime

    input (torch array)

    return output (torch array)
    """

    inputs = {self.session.get_inputs()[i].name: OnnxAdapter.to_numpy(arg) for i, arg in enumerate(input)}
    outputs = self.session.run(None, inputs)
    # convert the output back to torch tensors
    outputs = [torch.from_numpy(o) for o in outputs]
    if len(outputs) > 1:
      return outputs
    else:
      return outputs[0]

  @staticmethod
  def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()