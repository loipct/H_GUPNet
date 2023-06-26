## Introduction 
Graduation thesis : Monocular 3D Object Detection with Deep Learning  (Ho Chi Minh City University of Technology, HCMUT)

Author : [Loi Nguyen](mailto:huuloi312001@gmail.com), [Dat Ngo](mailto:phatdat906@gmail.com)

We carefully analyze and study the [GUPNet](https://arxiv.org/abs/2107.13774) base design; then proposed H_GUPNet, a monocular 3D object detection framework to enhance the object detection performance of the underlying design.

## H_GUPNet

H_GUPNet is a Monocular 3D Object Detection framework based on the [GUPNet](https://arxiv.org/abs/2107.13774) base design combined with the [Homography loss](https://arxiv.org/abs/2204.00754) function to enhance the object detection performance of the underlying design. Most current testing is done on single stage detectors that's why we brought this loss function to two stage detector - [GUPNet](https://arxiv.org/abs/2107.13774).

## Other relative things
1. The releases code is originally set to train on multi-category here. If you would like to train on the single category (Car), please modify the code/experiments/config.yaml. Single-category training can lead to higher performance on the Car. 

2. This implementation includes some tricks that do not describe in the paper. Please feel free to ask me in the issue. And I will also update the principle of them in the [supplementary materials](https://github.com/loiprocute/H_GUPNet/blob/main/pdf/supp.pdf)

3. The overall code cannot completely remove randomness because we use some functions which do not have reproduced implementation (e.g. ROI align). So the performance may have a certain degree of jitter, which is normal for this project. 

## Contact

If you have any question about this project, please feel free to contact huuloi312001@gmail.com or phatdat906@gmail.com

## Acknowledgements
This code benefits from the excellent works: [GUPNet](https://github.com/SuperMHP/GUPNet).
