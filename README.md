# HyperFlow

## Base code
As a base code for our implementation we used [PointFlow](https://arxiv.org/abs/1906.12320) implementation published [here](https://github.com/stevenygd/PointFlow).

## Dependencies
* Python 3.6
* CUDA 10.0.
* G++ or GCC 5.
* [PyTorch](http://pytorch.org/). Codes are tested with version 1.0.1
* [torchdiffeq](https://github.com/rtqichen/torchdiffeq).

Following is the suggested way to install these dependencies: 
```bash
# Create a new conda environment
conda create -n HyperFlow python=3.6
conda activate HyperFlow

# Install pytorch (please refer to the commend in the official website)
conda install pytorch=1.0.1 torchvision cudatoolkit=10.0 -c pytorch -y

# Install other dependencies such as torchdiffeq, structural losses, etc.
./install.sh
```
## Dataset 

The point clouds are uniformly sampled from meshes from ShapeNetCore dataset (version 2) and use the official split.
Please use this [link](https://drive.google.com/drive/folders/1G0rf-6HSHoTll6aH7voh-dXj6hCRhSAQ?usp=sharing) to download the ShapeNet point cloud.
The point cloud should be placed into `data` directory.
```bash
mv ShapeNetCore.v2.PC15k.zip data/
cd data
unzip ShapeNetCore.v2.PC15k.zip
```

## Training

Example of training script: 
```bash
# Training setting for single airplane class. For other classes just chagne --cates parameter
./train.sh 
```

## Pre-trained model and test

Pretrained models are located in `pretrained_model` folder. 
```bash

# Evaluate the generative performance of HyperFlow trained on the airplane, car and chair categories for various variances (variance equal 0 reffers to mesh representation).
CUDA_VISIBLE_DEVICES=0 ./run_test_airplane.sh
CUDA_VISIBLE_DEVICES=0 ./run_test_car.sh
CUDA_VISIBLE_DEVICES=0 ./run_test_chair.sh
```

## Demo

The demo relies on [Open3D](http://www.open3d.org/). The following is the suggested way to install it:
```bash
conda install -c open3d-admin open3d 
```
The demo will sample shapes from a pre-trained model, save those shapes under the `supplementary` folder, and visualize
meshes, point clouds, and raw data for point clouds. Meshes are generated with various radius sizes. There 6 demonstration options
controlled by `--demo_mode` argument: 

0 - visualizing steps for transforming surface to mesh

1 - visualizing interpolations between two meshes

2 - visualizing generated meshes

3 - visualizing steps for transforming samples from logNormal to point cloud

4 - visualizing interpolations between two point clouds

5 - visualizing generated point clouds from logNormal

Once this dependency is in place, you can use the following script to use the demo for the pre-trained model for airplanes:
```bash
CUDA_VISIBLE_DEVICES=0 ./run_demo_airplane.sh
```
