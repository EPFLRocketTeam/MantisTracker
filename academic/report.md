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

Unquestionably, space agencies around the world develop their own tracker for their space vehicles. However, they use state-of-the-art and extremely costly hardware in order to track the device at very high altitude and thus are out of our scope of research. Moreover, none of them publish the software and the methods used in the trackers. For further information, you can refer to European Space Agency (ESA) and their Estrack system[1] which is one of the rare resource given by a space agency. 

In Europe, a network of amateur rocketery associations is well developed. The goal is to build sounding rockets using the knowledge or skills of students and space enthusiasts. A international competition, Euroc[2], gather them for launching and comparing the different aerial vehicles. Unfortunately, none of these associations publish their work on a custom rocket tracker. There might be two reasons for that: either they don't build one or they're keeping the tracker proprietary. For instance, Wüspace [3] is a german association that partly focus on conceiving a tracker, but their project is not open-source.

Fortunately, object detection and tracking is a buzzing field in computer vision. The research is progressing at a fast pace with new methods emerging every year. Many detection and tracking algorithms will be presented in the following sections. Concerning public research on the specific challenge of rocket tracking, not much has been published in the litterature. Many publications tackle the tracking question by using radars and signal processing. The paper "Kalman Filter Embedded in FPGA to Improve Tracking Performance in Ballistic Rockets"[4] from 2013 analyses the possibility to use radars and Kalman filters to process the signal and predict the rocket trajectory. Following the same idea, "Long range radio location system employing autonomous tracking for sounding rocket"[5] is a more recent work. The puplication presents a custom signal processing algorithm that controls a movable antenna. Despite those prior researches, we decided to rely on a camera and video processing instead of antennas. Our goal is to capture a video footage of the rocket launch and therefore image quality and framing must be optimized. Besides, the cost of capable radars or antennas is higher than the one of simple cameras and computers. In 2017, researchers explored this idea in "Rocket launch detection and tracking using EO sensor" in which they display how to process a video footage to track missiles and rockets for a military purpose.

[1]https://www.esa.int/Enabling_Support/Operations/ESA_Ground_Stations/Estrack_ground_stations
[2]https://euroc.pt
[3]https://www.wuespace.de
[4]
[5]

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