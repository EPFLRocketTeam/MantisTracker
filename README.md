# MantisTracker

Rocket Tracker: follow and capture your rocket when launching.

The tracker detects the rocket on a live camera feed and adjust its stand to follow it. 

## Structure

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

## Usage

`pipeline.py` is the entry point of the tracker. Run `python3 pipeline.py` with the following parameters:

- `--benchmark`: use this option with any method to run a benchmark that returns the speed and the accuracy of the run.

- `detect`: detect a rocket on the provided images

  - `--images`: path to the images to use in the detection (required).
  - `--method`: specify the method of detection (default: Yolo). The options are [Yolo, tiny-Yolo].

- `track`: track a rocket on a provided video feed.

  - `--video`: the video feed. Can be a path to a video or TODO
  - `--method`: specify the method of tracking (default: OpenTracker). The options are [MOSSE; KCF, SiamRPN].
  - `--detection`: specify the method of detection (default: Yolo). The options are [Yolo, tiny-Yolo].

  ## Installation on Nvidia Jetson Nano

  - `OpenCV`, `numpy`, `pyyaml` using pip
  - `Pytorch`: https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-7-0-now-available/72048
  - `ONNX runtime`: https://developer.nvidia.com/blog/announcing-onnx-runtime-for-jetson/
  
## Model ZOO

Weights can be downloaded there: [Model ZOO](https://drive.google.com/drive/folders/107ANqfyJynHCv95W-yuA8TASNIbiQ2vz?usp=sharing)

Two models are available for YOLO and Tiny YOLO which correspond to the two trainings done with different datasets. Second is best.
