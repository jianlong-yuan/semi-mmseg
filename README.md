# Semi-supervised Semantic Segmentation with Mutual Knowledge Distillation

This project hosts the codo for implementing the MKD algorithm for semi-supervised learning


# Installation

## Requirements

- Linux (Windows is not officially supported)
- Python 3.6+
- PyTorch 1.10 or higher
- CUDA 9.0 or higher
- GCC 4.9 or higher
- mmcv-full


### Install MKD

a. Create a conda virtual environment and activate it.

```shell
conda create -n mkd python=3.6 -y
conda activate mkd
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch=1.10.0 torchvision cudatoolkit=10.2 -c pytorch -y
```

c. Install mmcv-full.

```shell
pip install mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
```

d. Install other third-party libraries.

```shell
pip install terminaltables imgaug onnxruntime==1.6.0 onnx albumentations Scikit-Image pycocotools tensorboard pillow==8.4.0
```

e. Clone the MKD repository.

```shell
git clone https://github.com/jianlong-yuan/semi-mmseg.git
cd semi-mmseg
```

f. Install.

```shell
pip install -r requirements.txt
pip install -e .  # or "python setup.py develop"
```

### Prepare datasets

It is recommended to symlink your dataset root (assuming `YOUR_DATA_ROOT`) to `$semi-mmseg/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

#### Prepare PASCAL VOC 2012 and Cityscapes

Assuming that you usually store datasets in `$YOUR_DATA_ROOT` (e.g.,`/home/YOUR_NAME/data/`).

The different split lists will be store in data directory.

```
MKD
├── configs
├── data
│   ├── cityscapes
│   │   ├── images
│   │   ├── segmentation
│   |   ├── splits
│   │   |   |    ├── cps_splits
│   │   |   |    ├── u2pl_splits
│   ├── VOCdevkit
│   │   ├── VOC2012
│   │   |   ├── Annotations
│   │   |   ├── ImageSets
│   │   |   ├── JPEGImages
│   │   |   ├── SegmentationClass
│   │   |   ├── SegmentationClassAug
│   │   |   ├── SegmentationObject
│   │   |   ├── splits
│   │   |   |    ├── cps_splits
│   │   |   |    ├── pseudoseg_splits
│   │   |   |    ├── u2pl_splits
```



# Training

```./tools/dist_train.sh configs/semi_ablations/cps_meanteacher_3b_w1.5_w.1.0_fdmt1.5.py```

## Acknowledgement

We would like to thank the [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) for its open-source project.
