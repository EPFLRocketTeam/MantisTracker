import tensorrt as trt
import numpy as np
import pycuda.driver as cuda

# Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# Flag for the network
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

# TensorRT primitives
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(EXPLICIT_BATCH)

# Parser of the model
parser = trt.OnnxParser(network, TRT_LOGGER)

print("Parsing ONNX model...")

with open("tracking/training/onnx/backbone.onnx", 'rb') as model:
  succ = parser.parse(model.read()) 
  if not succ:
    for error in range(parser.num_errors):
          print(parser.get_error(error))

print("ONNX model parsed !")

print("Infering...")

# for inference
config = builder.create_builder_config()
engine = builder.build_engine(network, config)
context = engine.create_execution_context() 

print(context.all_binding_shapes_specified)

# This determines the amount of memory available to the builder when building an optimized engine 
# and should generally be set as high as possible.
config.max_workspace_size = 1 << 20 

# Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
# Allocate device memory for inputs and outputs.
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
# Create a stream in which to copy inputs/outputs and run inference.
stream = cuda.Stream()

# Transfer input data to the GPU.
cuda.memcpy_htod_async(d_input, h_input, stream)
# Run inference.
context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
# Transfer predictions back from the GPU.
cuda.memcpy_dtoh_async(h_output, d_output, stream)
# Synchronize the stream
stream.synchronize()
# Return the host output. 
#h_output

print("Inference done !")