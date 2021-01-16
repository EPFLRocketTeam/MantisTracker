import tensorrt as trt
import pycuda.driver as cuda

import torch
import numpy as np

class TensorRtAdapter:
  """
  An TensorRT adapter can be called to perform inference on a TensorRT engine.
  """

  def __init__(self, filename, cfx, stream):
    """
    Deserialize the TensorRT engine

    filename (string): path to the engine
    cfx: cuda context
    stream: queue for the jobs 
    """

    self.cfx = cfx
    self.stream = stream

    # Read the engine and create the execution context
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    with trt.Runtime(TRT_LOGGER) as runtime:
      with open(filename, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()

    # I/O information
    in_shape = self.context.get_binding_shape(0)
    in_size = trt.volume(in_shape) * np.dtype(np.float32).itemsize
    self.out_shape = self.context.get_binding_shape(1)

    # Host buffer for the ouput
    self.host_out = cuda.pagelocked_empty(trt.volume(self.out_shape), dtype=np.float32)

    # GPU buffer for the data
    self.device_in = cuda.mem_alloc(in_size)
    self.device_out = cuda.mem_alloc(self.host_out.nbytes)

  def __call__(self, input):
    """
    Infer using the TensorRT engine

    input (torch array)

    return output (torch array)
    """

    # New CUDA context
    self.cfx.push()

    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(self.device_in, TensorRtAdapter.to_numpy(input), self.stream)
    
    # Run inference.
    self.context.execute_async_v2(bindings=[int(self.device_in), int(self.device_out)], stream_handle=self.stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(self.host_out, self.device_out, self.stream)
    # Synchronize the stream
    self.stream.synchronize()
      
    # Remove CUDA context    
    self.cfx.pop()

    # The host_out is a flatten array
    return torch.from_numpy(self.host_out.reshape(self.out_shape))
  
  def __del__(self):
    self.cfx.pop()

  @staticmethod
  def to_numpy(tensor):
    array = tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    return np.ascontiguousarray(array)
