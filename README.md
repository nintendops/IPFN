<<<<<<< HEAD

# Implicit Periodic Field Network

This repository contains the code (in PyTorch) for [# IPFN: Exemplar-based Pattern Synthesis with Implicit Periodic Field Networks](https://arxiv.org/abs/2204.01671)  (CVPR'2022) by Haiwei Chen, Jiayi Liu, [Weikai Chen](http://chenweikai.github.io/), [Shichen Liu](https://shichenliu.github.io/), and [Yajie Zhao](https://www.yajie-zhao.com/).

## Introduction
![](https://github.com/nintendops/IPFN/blob/main/media/teaser.png)
IPFN is a fast, lightweighted generative network designed for synthesizing and expanding both stationary and directional visual patterns in both 2D and 3D.  Specifically, IPFN maps periodically encoded coordinates and a noise field to image pixels, signed distance, etc. and allows synthesis of visual patterns in an infinitely large space. 
![](https://github.com/nintendops/IPFN/blob/main/media/network.png)
Below are two animated examples that are synthesized in an expanding space:

Pebble             |  HoneyComb
:-------------------------:|:-------------------------:
![](https://github.com/nintendops/IPFN/blob/main/media/pebble_animated.gif)  |  ![](https://github.com/nintendops/IPFN/blob/main/media/comb_animated.gif)

[arXiv](https://arxiv.org/abs/2204.01671)  | [BibTex](#contact)


## Requirements

The code has been tested on Python3.7, PyTorch 1.7.1 and CUDA (10.1). The additional dependencies can be installed with 
```
python -m pip install -r requirement
```

## Experiments

**Datasets**
We provide several input images, sampled from the [Describable Textures Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/), in [the exemplar folder](https://github.com/nintendops/IPFN/blob/main/exemplars/images). 

**Demo**
We provide a simple walk through of how to train and generate texture images with our model in a [demo notebook](https://github.com/nintendops/IPFN/blob/main/demo.ipynb), where you may either choose to train a synthesizer from scratch or to utilize a pretrained model for synthesizing honeycomb image.

**Training**
Training on a single examplar input can be done with the following command:
```
# with visdom server
visdom -port 8097 &
python run.py experiment -n EXPERIMENT_NAME dataset -p PATH_TO_IMAGE visdom --port 8097
```
```
# without visdom server
python run.py experiment -n EXPERIMENT_NAME dataset -p PATH_TO_IMAGE visdom --display-id -1
```

In addition, depending on the task of interest, the `--input-type` arguments may be used to select whether the network is trained in 2D or 3D coordinate space; the `--guidance-feature-type` may be used to control synthesis of directional patterns.

## Contact

Haiwei Chen: chw9308@hotmail.com
Any discussions or concerns are welcomed!

**Citation**
If you find our project useful in your research, please consider citing:

```
@inproceedings{chen2022exemplar,
  title={Exemplar-Based Pattern Synthesis With Implicit Periodic Field Network},
  author={Chen, Haiwei and Liu, Jiayi and Chen, Weikai and Liu, Shichen and Zhao, Yajie},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3708--3717},
  year={2022}
}
```
=======
# IPFN
pytorch implementation for the paper [Exemplar-based Pattern Synthesis with Implicit Periodic Field Networks (CVPR 2022)](https://arxiv.org/abs/2204.01671) by Haiwei Chen, Jiayi Liu, [Weikai Chen](http://chenweikai.github.io/), [Shichen Liu](https://shichenliu.github.io/) and Yajie Zhao.

> **Note:** codes are functional, but I am still in the progress of adding actual information to this README file... (dependencies, code overview and some visuals)
>>>>>>> e41c42fa2d56db5b6aa0538b800dfa2c24eed8f0

