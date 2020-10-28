# Conceiving a rocket tracker

- Abstract
- Related Work
- Methodology
- Result
- Discussion
- Future work

## Abstract

The overall task aims to conceive a mechanism that can record the launch of a rocket from the ground station. It will be equipped with one or more cameras on a moving support controlled by a software. This project specifically designs the software. The program must implement an effecient real-time tracking algorithm to follow the rocket on the camera. It must also receives metrics from the rocket such as speed, altitude, etc. Using those information, the software moves the camera accordingly.

## Related work

- [WüSpace](https://www.wuespace.de) and the T-REX tracker: a student association based in Deutschland. They conceived a rocket tracker, but their project is not open source.


## Part 1: Computer vision

### Objectives and constraints

The tracking algorithm receives the live image from the camera and ouput the position of the rocket using a bounding box.
It must respect the following requirements:

- The tracking algorithm follows the rocket in the image.
- The tracking algorithm is robust against the noise in the image such as sun, clouds, birds, etc.
- The tracking algorithm is able to recover if the rocket is lost.
- The tracking algorithm is fast enough, namely acheive enough frame per second (fps), to follow the rocket when launching.
- The tracking algorithm can run on the given hardware that might be limited.

The software must alos implement an object detection algorithm to bound the rocket at the beginning. The object detection can also be used during the tracking to improve the result if necessary. 

### Finding a tracking algorithm

Computer Vision has been a trending field these past years and tracking methods are evolving quickly thanks to the research. Many different tracking technics exist and trackers derived from those are uncountable. The explosion of trackers makes the work of finding a suitable algorithm challeging. Therefore, we selected a few trackers from the disctinct subfields.

#### Correlation filters

- Simple and performant tracker, can be implement using only OpenCV.
- Problem: the tracker cannot recover if the object disappears, which is inherent to the technics used. This problem could be mitigated using the object detection in pair with the tracker.

#### Deep learning

##### Atom

- Takes the background into account which lead to better performance on non uniform background.
- Target estimation is offline but classification is online which require GPU hardware to be performant. 

##### Siamese Network

- Siamese networks are trained offline, pretrained models are available to use the tracker out of the box. The algorithm still need much more computing power in comparison to a CF algorithm.

### Finding an object detection algorithm

#### Yolo v4

- One stage detection, which are the more efficient algorithm but at the cost of precision.

#### EfficientNet

- Two stage detection

#### R-CNN family

- Two stage detection.


## Part 2: Control and move the camera

## Part 3: Implementation with the hardware

### Selecting the hardware

#### Computer

#### Camera