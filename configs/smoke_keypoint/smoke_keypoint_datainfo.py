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
