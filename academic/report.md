# Conceiving a rocket tracker

## Abstract

The overall task aims to conceive a mechanism that can record the launch of a rocket from the ground station. It will be equipped with one or more cameras on a moving support controlled by a software. This project specifically designs the software. The program must implement an effecient real-time tracking algorithm to follow the rocket on the camera. It must also receives metrics from the rocket such as speed, altitude, etc. Using those information, the software moves the camera accordingly.

## Introduction

## Related work

Unquestionably, space agencies around the world develop their own tracker for their space vehicles. However, they use state-of-the-art and extremely costly hardware in order to track the device at very high altitude and thus are out of our scope of research. Moreover, none of them publish the software and the methods used in the trackers. For further information, you can refer to European Space Agency (ESA) and their Estrack system[1] which is one of the rare resource given by a space agency. 

In Europe, a network of amateur rocketery associations is well developed. The goal is to build sounding rockets using the knowledge or skills of students and space enthusiasts. A international competition, Euroc[2], gather them for launching and comparing the different aerial vehicles. Unfortunately, none of these associations publish their work on a custom rocket tracker. There might be two reasons for that: either they don't build one or they're keeping the tracker proprietary. For instance, Wüspace [3] is a german association that partly focus on conceiving a tracker, but their project is not open-source.

Fortunately, object detection and tracking is a buzzing field in computer vision. The research is progressing at a fast pace with new methods emerging every year. Many detection and tracking algorithms will be presented in the following sections. Concerning public research on the specific challenge of rocket tracking, not much has been published in the litterature. Many publications tackle the tracking question by using radars and signal processing. The paper "Kalman Filter Embedded in FPGA to Improve Tracking Performance in Ballistic Rockets"[4] from 2013 analyses the possibility to use radars and Kalman filters to process the signal and predict the rocket trajectory. Following the same idea, "Long range radio location system employing autonomous tracking for sounding rocket"[5] is a more recent work. The puplication presents a custom signal processing algorithm that controls a movable antenna. Despite those prior researches, we decided to rely on a camera and video processing instead of antennas. Our goal is to capture a video footage of the rocket launch and therefore image quality and framing must be optimized. Besides, the cost of capable radars or antennas is higher than the one of simple cameras and computers. In 2017, researchers explored this idea in "Rocket launch detection and tracking using EO sensor" in which they display how to process a video footage to track missiles and rockets for a military purpose. Their approach is to track the rocket using feature matching with the ORB algorithm. They specifically detect the nosetip of the rocket and track it across the video frames.

## Methodology

### Objectives and constraints

The tracker software receives the live image from the camera and ouput the position of the rocket using a bounding box.
It must respect the following requirements:

- The tracker follows the rocket in the image.
- The tracker is robust against the noise in the image such as sun, clouds, birds, etc.
- The tracker is able to recover if the rocket is lost.
- The tracker is fast enough, namely acheive enough frame per second (fps), to follow the rocket when launching.
- The tracker can run on the given hardware that might be limited.

The software uses a combination of object detection and tracking methods to achieve the best accuracy and speed possible given the hardware limitations.

### Finding an object detection algorithm

The object detection's goal is to find and delimit a given object on an image. For each object one need to detect, the framework must be trained with an annotated set of object's images. The annotation process consists of drawing bounding boxes around the object and label them.

In the past years, the research on object detector has been very active and thanks to that interest, the choices are numerous. The detection methods improve their accuracy constantly in a though competition, but the accuray often comes at a performance cost. Indeed, the best methods are often very deep networks with many layers and weights that need a powerful GPU in order to run in real time. YOLO is a famous object detector existing since 2016 and being at its fifth generation (YOLO v5). YOLO uses a one-pass method which predict bounding boxes and assign class probabilities with a single convolutional network. Previous methods like R-CNN would have a two-pass algorithm: first a region proposal and then a classifier for the proposed boxes. YOLO and its approach enables a similar accuracy while boosting its speed. We choose to train a YOLO detector that would detect sounding rocket and that could be use for two purposes in the project:

- The initial detection in the video to start tracking the rocket
- Recover the rocket if the tracker fails to track it

## Training

We choose to train the YOLOv4 detector which was published in 2020[6]. The implemention is found the darknet repository [7]. The most challenging task is the creation of the training dataset because it will determine the accuracy of detection in the real world setting. In our case, the dataset is particularly difficult to gather because the object is not common. From the start, we knew that it would impossible to collect thousands of sounding rocket's pictures and that our dataset would be rather houndreds of images. Therefore, the images should be carefully chosen to reflect every aspect of real world situation in which the tracker would operate. Thankfully, the EPFL Rocket Team maintain an archive of pictures and videos through the years and it contains footages of launches and rockets in-flight. 

We seeked through the content and selected 113 training pictures for our first training of YOLO. We also selected 17 images to form a testing dataset. The configuration and the dataset can be found in our Github repository. Here are the results. The algorithm converged with a mean average precision (mAP) of 96%. On our test dataset, rockets could be detected on 9 images out of 17, resulting in a 52% accuracy. We draw the following conclusions from the first training. First, the chosen YOLO convolutional network is complex enough to suit our pictures, which means we do not need to head for a more complex detector or backbone. On the other hand, the training dataset does not reflect all situations a rocket can be found in, explaining the poor result on the testing dataset.

We analyzed the successes and the failures to extract which type of images we needed to get a complete coverage. We defined x categories:
- A rocket on its launch rails. The rocket is vertical, but there might be a lot of background noise (the rail, people, the landscape, etc).
- A rocket right after its take-off. This is the easiest part, as the rocket is still vertical, but the background, the sky, tends to be simpler. The difficulty comes from the smoke the rocket emits.
- End of flight. The rocket is far away in the sky and is reaching its maximum altitude. Thus, the rocket becomes smaller and is not necessarly vertical. The bounding boxes are rectangles, such that when a rocket is skew the ratio background rocket is very high. In this phase, the tracker will eventually lose the rocket.

From

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

### Hardware

The tracker software is embeded on a portable tripod that can be deployed anywhere for a rocket launch. Therefore, the computer is powered by a battery, limiting the range of possible hardware to run the program. 

## Result

## Discussion

## Future work

[1]https://www.esa.int/Enabling_Support/Operations/ESA_Ground_Stations/Estrack_ground_stations
[2]https://euroc.pt
[3]https://www.wuespace.de
[4]
[5]
[6]https://arxiv.org/abs/2004.10934
[7]https://github.com/AlexeyAB/darknet/