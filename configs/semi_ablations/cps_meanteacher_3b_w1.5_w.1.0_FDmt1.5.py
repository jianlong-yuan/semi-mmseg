_base_ = [
    '../_base_/default_runtime.py',
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True, momentum=0.001)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 2),
        strides=(1, 2, 2, 1),
        # multi_grid=(1, 2, 4),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=256,
        dilations=(1, 6, 12, 18),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=-1.0,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, avg_non_ignore=True,)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# fp16 placeholder
fp16 = dict()
# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0005)
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=30000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True, start=20000)


algorithm = dict(
    type='ExpGeneralSemiCPSFDOnlineEMAOnce3CutmixSWFD',
    architecture=dict(
        type='MMSegArchitecture',
        model="same"),
    end_momentum=0.6,
    components=[
        dict(module='decode_head.conv_seg',
            losses=[
                dict(
                    type='ExpSemiLossCPSFAWS3',
                    name='loss_semi',
                    loss_weight=1.5,
                    loss_weight2=1.0,
                    avg_non_ignore=True,
                    ignore_index=255,
                    total_iteration=runner['max_iters'],
                    align_corners=False,
                    branch1=True,
                    branch2=True,
                    teacher1=True,
                    teacher2=True,
                    end_ratio=1.5,
                    scale_factor=0.5
                    )
            ])
    ]
)


# dataset settings
dataset_type = 'PascalVOCDataset'
data_root = 'data/VOCdevkit/VOC2012'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
img_scale = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='GenerateCutBox', prop_range=[0.25, 0.5], n_boxes=3, crop_size=crop_size),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'cutmask']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

train_sup=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClassAug',
        split = '/home/gongyuan.yjl/workspaces/segmentation/semi-mmseg/data/VOCdevkit/subset_train_aug/train_aug_labeled_1-16.txt',
        pipeline=train_pipeline)

unsup_train_pipeline = [
    dict(type='StrongWeakAug',
         pre_transforms=[
             dict(type='LoadImageFromFile'),
             dict(type='LoadAnnotations'),
             dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
             dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
             dict(type='RandomFlip', prob=0.5)],
         weak_transforms=[
            dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='GenerateCutBox', prop_range=[0.25, 0.5], n_boxes=3, crop_size=crop_size),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg', 'cutmask']),
         ],
         strong_transforms=[
            dict(type='Albu', transforms=[dict(type='SomeOf', n=1, p=1.0,
                transforms=[
                    dict(type='ColorJitter', brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=0.3),
                    dict(type="GaussianBlur", blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.5),
                    dict(type="Equalize", p=0.1),
                    dict(type="Solarize", p=0.1),
                    dict(type="ToGray", p=0.5)]),]),
            # dict(type='RandomAppliedTrans', transforms=[dict(type='RGB2Gray')], p=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='GenerateCutBox', prop_range=[0.25, 0.5], n_boxes=3, crop_size=crop_size),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg', 'cutmask']),
         ],)
]

train_unsup=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClassAug',
        split = '/home/gongyuan.yjl/workspaces/segmentation/semi-mmseg/data/VOCdevkit/subset_train_aug/train_aug_unlabeled_1-16.txt',
        pipeline=unsup_train_pipeline)


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type="Semi",
        sup_dataset=train_sup,
        unsup_dataset=train_unsup
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline))
