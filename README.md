# What's this
Implementation of GoogLeNet-v3 [[1]][Paper] by chainer


# Dependencies

    git clone https://github.com/nutszebra/googlenet_v3.git
    cd googlenet_v3
    git submodule init
    git submodule update

# How to run
    python main.py -p ./ -g 0 


# Details about my implementation

* Data augmentation  
Train: Pictures are randomly resized in the range of [299, 512], then 299x299 patches are extracted randomly and are normalized locally. Horizontal flipping is applied with 0.5 probability.  
Test: Pictures are resized to 406x406, then they are normalized locally. Single image test is used to calculate total accuracy. 

* Auxiliary classifiers  
No implementation

* Gradient clipping  
2.0

* RMSprop  
decay is 0.9 and eps is 1.0 as [[1]][Paper] said.

* Learning rate  
Initial learning rate is 0.045 acoording to [[1]][Paper], and it is multiplied by 0.94 at every 2 epochs.

* Weight decay  
According to [[2]][Paper], weight decay is 4.0*10^-5.


# Cifar10 result

| network              | depth  | total accuracy (%) |
|:---------------------|--------|-------------------:|
| my implementation    | 49     | soon               |

<img src="https://github.com/nutszebra/googlenet_v3/blob/master/loss.jpg" alt="loss" title="loss">
<img src="https://github.com/nutszebra/googlenet_v3/blob/master/accuracy.jpg" alt="total accuracy" title="total accuracy">

# References
Rethinking the Inception Architecture for Computer Vision [[1]][Paper]  
Xception: Deep Learning with Depthwise Separable Convolutions [[2]][Paper1]  
[paper]: https://arxiv.org/abs/1512.00567 "Paper"
[paper1]: https://arxiv.org/pdf/1610.02357v2.pdf "Paper1"
