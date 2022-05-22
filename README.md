# ComGAN

This is the official code of our paper "ComGAN: Unsupervised Disentanglement and ﻿Segmentation via Image Composition"

## Requirements

- easydict==1.9
- matplotlib==2.2.3
- Pillow==5.2.0
- torch==1.8.0+cu111
- torchvision==0.9.0+cu111
- numpydoc==0.8.0

## Getting started

### Setting up the data

**Note**: You need to download the data if you wish to train your own model.

Download the formatted CUB data from this [link](https://pan.baidu.com/s/1-HGgdQ3dXwvWfXJq1xMwhQ )[BaiDuYunDisk] and its extracted code: 2dc3  and extract it inside the `data` directory

```shell
cd data
unzip birds.zip
cd ..
```

### Downloading pre-trained models

Pretrained generator models for CUB are available at this [link](https://pan.baidu.com/s/1a7Qyy2vvlK4r7iEgBXd2yA )[BaiDuYunDisk] and its extracted code:xcl3. Download and extract them in the `models_pth/birds` directory.

## Evaluating the model

### The disentanglement module

In `code/config.py`:

- Specify the data path in `DATA_DIR`.
- Specify the generator path in `TEST.NET_G`
- Specify the output directory to save the generated images in `SAVE_DIR`.
- Run `python disentanglement_module.py SAVE_DIR`

### The segmentation module

In `code/config.py`:

- Specify the segmentation model path in `TEST.NET_U`
- Run `python segmentation_module.py SAVE_DIR`

## Training your own model

In `code/config.py`:

- Specify the dataset location in `DATA_DIR`.
- Specify the dimension of variables that you wish for DS-ComGAN, in `FINE_GRAINED_CATEGORIES`.
- Specify the training hyperparameters in `TRAIN.FLAG`.
- Run `python disentanglement_module.py SAVE_DIR`
- Run `python segmentation_module.py SAVE_DIR`

## Acknowledgement

We thank the authors of [FineGAN: Unsupervised Hierarchical Disentanglement for Fine-Grained Object Generation and Discovery ](https://arxiv.org/abs/1811.11155) for releasing their source code.



