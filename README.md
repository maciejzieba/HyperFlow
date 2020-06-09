# HyperFlow

## Base code
As a base code for our implementation we used [PointFlow](https://arxiv.org/abs/1906.12320) published [here](https://github.com/stevenygd/PointFlow).

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

Example training script: 
```bash
.train.sh 
```

## Pre-trained model and test

Pretrained models can be downloaded from this [link](https://drive.google.com/file/d/1dcxjuuKiAXZxhiyWD_o_7Owx8Y3FbRHG/view?usp=sharing). 
The following is the suggested way to evaluate the performance of the pre-trained models.
```bash
unzip pretrained_models.zip;  # This will create a folder named pretrained_models

# Evaluate the reconstruction performance of an AE trained on the airplane category
CUDA_VISIBLE_DEVICES=0 ./scripts/shapenet_airplane_ae_test.sh; 

# Evaluate the reconstruction performance of an AE trained with the whole ShapeNet
CUDA_VISIBLE_DEVICES=0 ./scripts/shapenet_all_ae_test.sh;

# Evaluate the generative performance of PointFlow trained on the airplane category.
CUDA_VISIBLE_DEVICES=0 ./scripts/shapenet_airplane_gen_test.sh
```

## Demo

The demo relies on [Open3D](http://www.open3d.org/). The following is the suggested way to install it:
```bash
conda install -c open3d-admin open3d 
```
The demo will sample shapes from a pre-trained model, save those shapes under the `demo` folder, and visualize those point clouds.
Once this dependency is in place, you can use the following script to use the demo for the pre-trained model for airplanes:
```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/shapenet_airplane_demo.py
```
