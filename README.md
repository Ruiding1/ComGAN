# ComGAN (NeurIPS 2022)

This is the official code of our paper "ComGAN: Unsupervised Disentanglement and Segmentation via Image Composition" [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1df282080150537df7b00c20aadcafad-Abstract-Conference.html)

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

Download the formatted CUB data from this [link](https://pan.baidu.com/s/1Ey7fZ1Bs_OLHy3-J7PTTeA )[BaiDuYunDisk] and its extracted code: 2dc3  and extract it inside the `data` directory

```shell
cd data
unzip birds.zip
cd ..
```

### Downloading pre-trained models

Pretrained generator models for CUB are available at this [link](https://pan.baidu.com/s/1GkzRIjz25vEs6uQ1hjcjaA )[BaiDuYunDisk] and its extracted code:xcl3. Download and extract them in the `models_pth/birds` directory.

## Evaluating the model

### The disentanglement module

In `code/config.py`:

- Specify the data path in `DATA_DIR`.
- Specify the generator path in `TEST.NET_G`
- Specify the output directory to save the generated images in `SAVE_DIR`.
- Run `python disentanglement_module.py SAVE_DIR`

### The segmentation module

In `code/config.py`:

- Specify the data path in `DATA_DIR`.
- Specify the generator path in `TEST.NET_G`

- Specify the segmentation model path in `TEST.NET_U`
- Run `python segmentation_module.py SAVE_DIR`

## Training your own model

In `code/config.py`:

- Specify the dataset location in `DATA_DIR`.
- Specify the dimension of variables that you wish for DS-ComGAN, in `FINE_GRAINED_CATEGORIES`.
- Specify the training switch in `TRAIN.FLAG`.
- Run `python disentanglement_module.py SAVE_DIR`
- Run `python segmentation_module.py SAVE_DIR`

 ## Bibtex
```
@inproceedings{ding2022comgan,
  title={ComGAN: unsupervised disentanglement and segmentation via image composition},
  author={Ding, Rui and Guo, Kehua and Zhu, Xiangyuan and Wu, Zheng and Wang, Liwei},
  booktitle={Advances in neural information processing systems},
  volume={35},
  pages={4638--4651},
  year={2022}
}
```

## Acknowledgement

We thank the following authors for releasing their source code, data and models:

- [FineGAN: Unsupervised Hierarchical Disentanglement for Fine-Grained Object Generation and Discovery ](https://arxiv.org/abs/1811.11155).
- [Unsupervised Foreground Extraction via Deep Region Competitio](https://arxiv.org/abs/2110.15497#).
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).
