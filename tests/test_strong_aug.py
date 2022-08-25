# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import mmcv
import numpy as np
import pytest
from mmcv.utils import build_from_cfg
from PIL import Image
from tqdm import tqdm
from mmseg.datasets.builder import PIPELINES
from mmseg.models.semi.datasets.pipelines import *
from mmseg.datasets.pipelines.compose import Compose

def test():
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    crop_size = (512, 512)
    train_pipeline = [
        # dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
        # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        # dict(type='RandomFlip', prob=0.5),
        # dict(type='CLAHE'),
        # dict(type='RGB2Gray'),
        # dict(type='RandomCutOut', prob=1.0, n_holes=(2, 5), cutout_ratio=(0.1, 0.3)),
        # dict(type='PhotoMetricDistortion'),
        # dict(type='Albu', transforms=[dict(type='SomeOf', n=1, p=1.0,
        #         transforms=[
        #             dict(type='ColorJitter', brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=0.5),
        #             dict(type="GaussianBlur", blur_limit=(3, 5), sigma_limit=(0.1, 2.0), p=0.7),
        #             dict(type="Equalize", p=0.5),
        #             dict(type="Solarize", p=0.2),
        #             dict(type="RandomBrightnessContrast", p=0.5),
        #             dict(type="GaussNoise", p=0.7),
        #             dict(type="ToGray", p=0.5)
        #             ]),
        #             ])
        # dict(type='RandomAppliedTrans', transforms=[dict(type='RGB2Gray')], p=0.5),
        # dict(type='PhotoMetricDistortion'),
        dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='copypaste',
             pre_transforms = [dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),],
             mix_pre_transforms=[dict(type='LoadImageFromFile'),
                                 dict(type='LoadAnnotations'),
                                 dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
                                 dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
                                 dict(type='RandomFlip', prob=0.5),
                                 dict(type='PhotoMetricDistortion'),
                                 dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
                                 ],
             prob=1., num_imgs=2, img_scale=(512, 512),
             num_object=1, exclude=[0, 255]),
        # dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    ]
    transforms = Compose(train_pipeline)
    # transforms = [build_from_cfg(t, PIPELINES) for t in train_pipeline]
    results = dict()
    img = mmcv.imread("/home/gongyuan.yjl/workspaces/segmentation/semi-mmseg/data/VOCdevkit/VOC2012/JPEGImages/2007_000333.jpg", 'color')
    h, w, _ = img.shape
    seg = np.array(Image.open(osp.join("/home/gongyuan.yjl/workspaces/segmentation/semi-mmseg/data/VOCdevkit/VOC2012/SegmentationClass/2007_000333.png")))
    results['img'] = img
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    # results['seg_fields'] = []
    results['img_prefix'] = "../data/VOCdevkit/VOC2012/JPEGImages/"
    results['seg_prefix'] = "../data/VOCdevkit/VOC2012/SegmentationClass/"
    results['img_infos'] = [dict(filename="2011_002717.jpg", ann=dict(seg_map="2011_002717.png")),
                            dict(filename="2011_002752.jpg", ann=dict(seg_map="2011_002752.png")),
                            dict(filename="2011_002767.jpg", ann=dict(seg_map="2011_002767.png")),
                            dict(filename="2011_002770.jpg", ann=dict(seg_map="2011_002770.png")),
                            dict(filename="2011_002834.jpg", ann=dict(seg_map="2011_002834.png")),
                            ]
    # {'filename': '2009_002567.jpg', 'ann': {'seg_map': '2009_002567.png'}}
    # {'img_info': {'filename': '2009_002567.jpg', 'ann': {'seg_map': '2009_002567.png'}},
    #  'ann_info': {'seg_map': '2009_002567.png'}, 'img_prefix': 'data/VOCdevkit/VOC2012/JPEGImages',
    #  'seg_prefix': 'data/VOCdevkit/VOC2012/SegmentationClass', 'seg_fields': []}
    # results['img_shape'] = img.shape
    # results['ori_shape'] = img.shape
    for i in tqdm(range(20)):
        results_out = transforms(copy.deepcopy(results))
        mmcv.imwrite(results_out['img'], "tmp2/{}.jpg".format(i))
        mmcv.imwrite(results_out['gt_semantic_seg'], "tmp2/{}.png".format(i))

if __name__ == "__main__":
    test()