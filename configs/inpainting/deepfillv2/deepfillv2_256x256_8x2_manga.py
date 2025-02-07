model = dict(
    type='TwoStageInpaintor',
    disc_input_with_mask=True,
    encdec=dict(
        type='DeepFillEncoderDecoder',
        stage1=dict(
            type='GLEncoderDecoder',
            encoder=dict(
                type='DeepFillEncoder',
                conv_type='gated_conv',
                channel_factor=0.75,
                padding_mode='reflect'),
            decoder=dict(
                type='DeepFillDecoder',
                conv_type='gated_conv',
                in_channels=96,
                channel_factor=0.75,
                out_act_cfg=dict(type='Tanh'),
                padding_mode='reflect'),
            dilation_neck=dict(
                type='GLDilationNeck',
                in_channels=96,
                conv_type='gated_conv',
                act_cfg=dict(type='ELU'),
                padding_mode='reflect')),
        stage2=dict(
            type='DeepFillRefiner',
            encoder_attention=dict(
                type='DeepFillEncoder',
                encoder_type='stage2_attention',
                conv_type='gated_conv',
                channel_factor=0.75,
                padding_mode='reflect'),
            encoder_conv=dict(
                type='DeepFillEncoder',
                encoder_type='stage2_conv',
                conv_type='gated_conv',
                channel_factor=0.75,
                padding_mode='reflect'),
            dilation_neck=dict(
                type='GLDilationNeck',
                in_channels=96,
                conv_type='gated_conv',
                act_cfg=dict(type='ELU'),
                padding_mode='reflect'),
            contextual_attention=dict(
                type='ContextualAttentionNeck',
                in_channels=96,
                conv_type='gated_conv',
                padding_mode='reflect'),
            decoder=dict(
                type='DeepFillDecoder',
                in_channels=192,
                conv_type='gated_conv',
                out_act_cfg=dict(type='Tanh'),
                padding_mode='reflect'))),
    disc=dict(
        type='MultiLayerDiscriminator',
        in_channels=4,
        max_channels=256,
        fc_in_channels=None,
        num_convs=6,
        norm_cfg=None,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
        out_act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
        with_spectral_norm=True,
    ),
    stage1_loss_type=('loss_l1_hole', 'loss_l1_valid'),
    stage2_loss_type=('loss_l1_hole', 'loss_l1_valid', 'loss_gan'),
    loss_gan=dict(
        type='GANLoss',
        gan_type='hinge',
        loss_weight=0.1,
    ),
    loss_l1_hole=dict(
        type='L1Loss',
        loss_weight=1.0,
    ),
    loss_l1_valid=dict(
        type='L1Loss',
        loss_weight=1.0,
    ),
    pretrained=None)

train_cfg = dict(disc_step=1)
test_cfg = dict(metrics=['l1', 'psnr', 'ssim'])

dataset_type = 'ImgInpaintingDataset'
input_shape = (256, 256)

train_pipeline = [
    dict(type='LoadImageFromFile', key='gt_img'),
    dict(
        type='LoadMask',
        mask_mode='irregular',
        mask_config=dict(
            num_vertices=(4, 10),
            max_angle=6.0,
            length_range=(20, 128),
            brush_width=(10, 45),
            area_ratio_range=(0.15, 0.65),
            img_shape=input_shape)),
    dict(
        type='Crop',
        keys=['gt_img'],
        crop_size=(384, 384),
        random_crop=True,
    ),
    dict(
        type='Resize',
        keys=['gt_img'],
        scale=input_shape,
        keep_ratio=False,
    ),
    dict(
        type='Normalize',
        keys=['gt_img'],
        mean=[127.5] * 3,
        std=[127.5] * 3,
        to_rgb=False),
    dict(type='GetMaskedImage'),
    dict(
        type='Collect',
        keys=['gt_img', 'masked_img', 'mask'],
        meta_keys=['gt_img_path']),
    dict(type='ImageToTensor', keys=['gt_img', 'masked_img', 'mask'])
]

test_pipeline = train_pipeline

data_root = '../mmdetection/datasets/manga/'

data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=2, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'data/inpaint/train.txt',
        data_prefix=data_root + 'data/inpaint/pages',
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'data/inpaint/test.txt',
        data_prefix=data_root + 'data/inpaint/pages',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'data/inpaint/test.txt',
        data_prefix=data_root + 'data/inpaint/pages',
        pipeline=test_pipeline,
        test_mode=True))

optimizers = dict(
    generator=dict(type='Adam', lr=0.0001), disc=dict(type='Adam', lr=0.0001))

lr_config = dict(policy='Fixed', by_epoch=False)

eval_every = 2000 # 50000
checkpoint_config = dict(by_epoch=False, interval=eval_every)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit'))
    ])

visual_config = dict(
    type='VisualizationHook',
    output_dir='visual',
    interval=1000,
    res_name_list=[
        'gt_img', 'masked_img', 'stage1_fake_res', 'stage1_fake_img',
        'stage2_fake_res', 'stage2_fake_img', 'fake_gt_local'
    ],
)

evaluation = dict(interval=eval_every)

total_iters = 1000003
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/test_pggan_manga'
load_from = './checkpoints/deepfillv2_256x256_8x2_places_20200619-10d15793.pth'
resume_from = './work_dirs/test_pggan_manga/latest.pth'
workflow = [('train', 10000)]
exp_name = 'deepfillv2_256x256_8x2_places'
find_unused_parameters = False

# python tools/train.py configs/inpainting/deepfillv2/deepfillv2_256x256_8x2_manga.py
# python demo/inpainting_demo.py configs/inpainting/deepfillv2/deepfillv2_256x256_8x2_manga.py work_dirs/test_pggan_manga/iter_5000.pth ../mmdetection/datasets/manga/data/inpaint/pages/138688/017.jpg ../mmdetection/datasets/manga/data/inpaint/masks/138688/017.png  test_5000.png
# python demo/inpainting_demo.py configs/inpainting/deepfillv2/deepfillv2_256x256_8x2_manga.py work_dirs/test_pggan_manga/iter_10000.pth ../mmdetection/datasets/manga/data/inpaint/pages/138688/017.jpg ../mmdetection/datasets/manga/data/inpaint/masks/138688/017.png  test_10000.png
# python demo/inpainting_demo.py configs/inpainting/deepfillv2/deepfillv2_256x256_8x2_manga.py work_dirs/test_pggan_manga/iter_15000.pth ../mmdetection/datasets/manga/data/inpaint/pages/138688/017.jpg ../mmdetection/datasets/manga/data/inpaint/masks/138688/017.png  test_15000.png
# python demo/inpainting_demo.py configs/inpainting/deepfillv2/deepfillv2_256x256_8x2_manga.py work_dirs/test_pggan_manga/iter_20000.pth ../mmdetection/datasets/manga/data/inpaint/pages/138688/017.jpg ../mmdetection/datasets/manga/data/inpaint/masks/138688/017.png  test_20000.png
# python demo/inpainting_demo.py configs/inpainting/deepfillv2/deepfillv2_256x256_8x2_manga.py work_dirs/test_pggan_manga/iter_37000.pth ../mmdetection/datasets/manga/data/inpaint/pages/138688/017.jpg ../mmdetection/datasets/manga/data/inpaint/masks/138688/017.png  test_37000.png