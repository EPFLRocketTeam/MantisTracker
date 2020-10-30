# Object tracking

## OpenCV

OpenCV offers many built-in tracking algorithms. They are based on filters and thus are simple and effecient. However, the algorithm can be deceiving in challeging situations.

Reference: [OpenCV documentation](https://docs.opencv.org/3.4/d9/df8/group__tracking.html)

## SiamFC family

The SiamFC family uses deep learning networks to track objects. In particular, Siamese convolution networks are commonly used in those methods. The project Pysot provides an implementation of SiamMask, SiamRPN++, DaSiamRPN, SiamRPN and SiamFC.

Reference: [Pysot](https://github.com/STVIR/pysot)