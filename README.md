# MantisTracker

Rocket Tracker: follow and capture your rocket when launching.

The tracker detects the rocket on a camera feed and adjust its movement mechanism to follow it. 

## Requirement

TODO

## Usage

`pipeline.py` is the entry point of the tracker. Run `python3 pipeline.py` with the following parameters:

- `--benchmark`: use this option with any method to run a benchmark that returns the speed and the accuracy of the run.

- `detect`: detect a rocket on the provided images

  - `--image`: path to the images to use in the detection (required).
  - `--method`: specify the method of detection (default: Yolo). The options are [Yolo, tiny-Yolo].

- `track`: track a rocket on a provided video feed.

  - `--video`: the video feed. Can be a path to a video or TODO
  - `--method`: specify the method of tracking (default: OpenTracker). The options are [OpenTracker, SiamRPN].
  - `--detection`: specify the method of detection (default: Yolo). The options are [Yolo, tiny-Yolo].