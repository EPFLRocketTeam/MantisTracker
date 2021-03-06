# MantisTracker

Rocket Tracker: follow and capture your rocket when launching.

The tracker detects the rocket on a live camera feed and adjust its stand to follow it. 

## Structure

- `/academic`: Bachelor report and videos which compare trackers.
- `/libraries`: Externel libraries used in the project. The references are in the folder.
- `/outputs`: Images generated by the detector/tracker.
- `/src`: Implementation of the tracker software.
- `/tools`: Entry points to run the services.
- `/training`: Models, config files, trained weights.

## Requirements

Install the following packages in order to run the tracker:

For all methods:
- `OpenCV` (>= 3.4): `pip install opencv-contrib-python`
- `numpy`: `pip insall numpy`
- `pyyaml` (>= 5.1): `pip install pyyaml`

If using conda: `conda create --name mantis -c anaconda opencv pyyaml numpy`

If you have GPU, ensure CUDA/cuDNN are properly installed. 

Additionally, if using conda: `conda install cudatoolkit=CUDA_VERSION cudnn`

For SiamRPN:

Please follow this link to install the requirements: [Pysot installation](https://github.com/STVIR/pysot/blob/master/INSTALL.md)

For ONNX:

- `ONNX runtime` : `pip install onnxruntime`

## Usage

Before using a detector, please download the weights and configure YOLO in the `config.yaml` file. 

For configuring SiamRPN, the config files are in `training/tracking/pysot/`. You can add `CUDA: False` to the config file to disable the use of a GPU.

`pipeline.py` is the entry point of the tracker. Run `python3 tools/pipeline.py` with the following parameters:

- `--benchmark`: use this option with any method to run a benchmark that returns the speed and the accuracy of the run.

- `detect`: detect a rocket on the provided images

  - `--images`: path to the images to use in the detection (required).
  - `--method`: specify the method of detection (default: Yolo). The options are [Yolo, tiny-Yolo].

- `track`: track a rocket on a provided video feed.

  - `--video`: the video feed. Can be a path to a video
  - `--method`: specify the method of tracking (default: OpenTracker). The options are [MOSSE, KCF, SiamRPN, SiamRPN-onnx].
  - `--detection`: specify the method of detection (default: Yolo). The options are [Yolo, tiny-Yolo].

Examples:
- `python3 tools/pipeline.py detect --images training/detection/dataset/test/*.jpg --method tiny-Yolo`
- `python3 tools/pipeline.py --benchmark track --video training/tracking/dataset/test/test2.mp4 --method SiamRPN`

The generated images of the detection or the tracking can be found in the folder `/outputs`.

### Export

`tools/export.py` is an experimental tool to export the SiamRPN tracker to ONNX and TensorRT.
- `python3 tools/export.py onnx`: export SiamRPN to ONNX models. Require onnx to be installed.
- `python3 tools/export.py tensorrt`: export ONNX models to TensorRT engines. Requires TensorRT to be installed.

## Installation on Nvidia Jetson Nano

- `OpenCV`, `numpy`, `pyyaml` using pip
- `Pytorch`: https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-7-0-now-available/72048
- `ONNX runtime`: https://developer.nvidia.com/blog/announcing-onnx-runtime-for-jetson/
  
## Model Zoo

Weights can be downloaded there: [Model Zoo](https://drive.google.com/drive/folders/107ANqfyJynHCv95W-yuA8TASNIbiQ2vz?usp=sharing)

Two models are available for YOLO and Tiny YOLO which correspond to the two trainings done with different datasets. Second is best.
