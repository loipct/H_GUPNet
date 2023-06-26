## Introduction 
<ins>Graduation thesis</ins> : "Monocular 3D Object Detection with Deep Learning"  

<ins>Author</ins> : [Loi Nguyen](mailto:huuloi312001@gmail.com), [Dat Ngo](mailto:phatdat906@gmail.com) - Ho Chi Minh City University of Technology, [HCMUT](https://en.wikipedia.org/wiki/Ho_Chi_Minh_City_University_of_Technology)

We carefully analyze and study the [GUPNet](https://arxiv.org/abs/2107.13774) base design; then proposed H_GUPNet, a monocular 3D object detection framework to enhance the object detection performance of the underlying design.

## H_GUPNet

H_GUPNet is a Monocular 3D Object Detection framework based on the [GUPNet](https://arxiv.org/abs/2107.13774) base design combined with the [Homography loss](https://arxiv.org/abs/2204.00754) function to enhance the object detection performance of the underlying design. Most current testing is done on single stage detectors that's why we brought this loss function to two stage detector - [GUPNet](https://arxiv.org/abs/2107.13774).


## Usage

### Train

Download the KITTI dataset from [KITTI website](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d), including left color images, camera calibration matrices and training labels.

Clone this project and then go to the code directory:

    git clone https://github.com/loiprocute/H_GUPNet.git
    cd code

You can build the environment easily by installing the requirements:

    conda env create -f requirements.yml
    conda activate gupnet

Train the model:

    CUDA_VISIBLE_DEVICES=0,1,2 python tools/train_val.py

### Evaluate

After training the model will directly feedback the detection files for evaluation (If so, you can skip this setep). But if you want to test a given checkpoint, you need to modify the "resume" of the "tester" in the code/experiments/config.yaml and then run:

    python tools/train_val.py -e

After that, please use the kitti evaluation devkit (deails can be refered to [FrustumPointNet](https://github.com/charlesq34/frustum-pointnets)) to evaluate:

    !g++ tools/kitti_eval/evaluate_object_3d_offline_apXX.cpp -o evaluate_object_3d_offline_apXX
    ./evaluate_object_3d_offline_apXX ../../Dataset/3d-detection/KITTI/training/label_2 ./outputs

We also provide the trained checkpoint which achieved the best multi-category performance on the validation set. It can be downloaded at [here](https://drive.google.com/file/d/1-iQEjNlWMGYC-wC4kN6We_TBbBmeKsmz/view?usp=sharing). This checkpoint performance is as follow:

<table align="center">
    <tr>
        <td rowspan="2",div align="center">Models</td>
        <td colspan="3",div align="center">Car@BEV IoU=0.7</td>    
        <td colspan="3",div align="center">Car@3D IoU=0.7</td>   
    </tr>
    <tr>
        <td div align="center">Easy</td> 
        <td div align="center">Mod</td> 
        <td div align="center">Hard</td> 
        <td div align="center">Easy</td> 
        <td div align="center">Mod</td> 
        <td div align="center">Hard</td> 
    </tr>
    <tr>
        <td div align="center">[GUPNet](https://arxiv.org/abs/2107.13774)</td>
        <td div align="center">31.07%</td> 
        <td div align="center">22.94%</td> 
        <td div align="center">19.75%</td> 
        <td div align="center">22.76%</td> 
        <td div align="center">16.46%</td> 
        <td div align="center">13.72%</td> 
    </tr>    
    <tr>
        <td div align="center">H_GUPNet</td>
        <td div align="center">31.96%</td> 
        <td div align="center">23.21%</td> 
        <td div align="center">20.07%</td> 
        <td div align="center">23.55%</td> 
        <td div align="center">16.58%</td> 
        <td div align="center">14.61%</td>  
    </tr>
</table>



## Other relative things
<ins>From Author for GUPNet</ins> :

1. The releases code is originally set to train on multi-category here. If you would like to train on the single category (Car), please modify the code/experiments/config.yaml. Single-category training can lead to higher performance on the Car. 
2. This implementation includes some tricks that do not describe in the paper. And We will also update the principle of them in the [supplementary materials](https://github.com/loiprocute/H_GUPNet/blob/main/pdf/supp.pdf)
3. The overall code cannot completely remove randomness because authors use some functions which do not have reproduced implementation (e.g. ROI align). So the performance may have a certain degree of jitter, which is normal for this project.
4. In the original paper, results on the val set are only training with the car category.
5. Mean moderate AP is about 15.5. So It is normal to achieve 15.5 moderate AP. 16.4 is the best one. On the validation set, authors run several times and select the best results, which are available for the val set but not suitable for the test set.
6. The training set of KITTI is small (about 3,000 cases), the results are not quite stable. When you utilize a larger dataset, it is easy to reproduce results because of better stability. For example, training with trainval (about 7000 cases) and evaluating on the test set. It is easy to get higher results than our reports without many times model selections.
7. Please mention that when evaluating on the test set, you should train on the trainval set (train set + eval set) rather than on train set. This setting is provided by previous works. So directly using checkpoint from train set is not right.
8. If your GPU cannot support 32 bsize, you can:
 - Set the bsize as big as possible.
 -  A trick may work but I cannot make sure. If your biggest bsize is b, you can re-scale the learning rate as lr*b/32. For example, your biggest bsize is 16 and our initial lr is 0.00125. You can set the lr as 0.00125*16/32.
9. The uncertainty loss can be negative.

<ins>Training time</ins>:

1. The training time per epoch of training on each class Car set is much higher than when training on 3 classes.
2. Training time per epoch when combined with Homography Loss is higher than when there is no Homography Loss.

## Contact

If you have any question about this project, please feel free to contact huuloi312001@gmail.com or phatdat906@gmail.com

## Acknowledgements
This code benefits from the excellent works: [GUPNet](https://github.com/SuperMHP/GUPNet).
