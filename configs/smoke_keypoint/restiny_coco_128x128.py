log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=1)
#evaluation = dict(interval=1000, metric='PCKh', save_best='PCKh')
evaluation = dict(interval=300, metric='mAP', save_best='AP')

optimizer = dict(
    type='Adam',
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 310
log_config = dict(
    interval=50, hooks=[
        dict(type='TextLoggerHook'),
    ])

numpoints = 3
channel_cfg = dict(
    num_output_channels=numpoints,
    dataset_joints=numpoints,
    dataset_channel=list(range(numpoints)),
    inference_channel=list(range(numpoints)))

# model settings
color_type = 'grayscale'
in_channels = 1
max_channels = 128
stem_channels = 16
base_channels = stem_channels

num_deconv_filters = (32,32)
num_deconv_kernels = (4,4)

model = dict(
    type='TopDown',
    backbone=dict(type='ResNet_Tiny', depth=14, max_channels=max_channels, stem_channels=stem_channels, base_channels=base_channels, in_channels=in_channels),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=128,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=2,
        num_deconv_filters=num_deconv_filters,
        num_deconv_kernels=num_deconv_kernels,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[128, 128],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file=None,
)

train_pipeline = [
    dict(type='LoadImageFromFile', color_type=color_type),
    dict(type='TopDownRandomFlipWithGrayscale', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='Cutout', shape_low=3, shape_high=30, pad_val_low=220, pad_val_high=255),
    dict(type='Sharpness', magnitude=0.5),
    dict(type='Brightness', magnitude=0.5),
    #dict(type='Contrast', magnitude=0.5),
    dict(type='ToTensor'),
    # dict(
    #     type='NormalizeTensor',
    #     mean=[0.5],
    #     std=[0.25]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'flip_pairs', 'bbox_score'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile', color_type=color_type),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    # dict(
    #     type='NormalizeTensor',
    #     mean=[0.5],
    #     std=[0.25]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['image_file', 'center', 'scale', 'rotation', 'flip_pairs', 'bbox_score']),
]

test_pipeline = val_pipeline

dataset_info = dict(
    dataset_name='smoke_keypoint',
    paper_info=dict(
        author='Streamax',
        title='2D smoke keypoints',
        container='W',
        year='2021',
        homepage='www.github.com',
    ),
    keypoint_info={
        0:
        dict(
            name='smoke_mouth',
            id=0,
            color=[255, 0, 0],
            type='lower',
            swap=''),
        1:
        dict(
            name='smoke_middle',
            id=1,
            color=[0, 255, 0],
            type='lower',
            swap=''),
        2:
        dict(
            name='smoke_end',
            id=2,
            color=[0, 0, 255],
            type='lower',
            swap=''),
    },
    skeleton_info={
        0:
        dict(link=('smoke_mouth', 'smoke_middle'), id=0, color=[255, 128, 0]),
        1:
        dict(link=('smoke_middle', 'smoke_end'), id=1, color=[255, 128, 0]),
    },
    joint_weights=[
        1,1,1
    ],
    # Adapted from COCO dataset.
    sigmas=[
        0.089, 0.083, 0.083
    ])

data_root = 'data/smoke_keypoint_coco'
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=64),
    train=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/train_coco.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info=dataset_info),
    val=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/val_coco.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info=dataset_info),
    test=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/val_coco.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info=dataset_info),
)
