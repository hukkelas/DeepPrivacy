import os
# Default values
_output_dir = "outputs"
_cache_dir = ".deep_privacy_cache"

model_size = 128
model_url = None
models = dict(
    max_imsize=128,
    min_imsize=8,
    pose_size=14,
    image_channels=3,
    conv_size={
        4: model_size,
        8: model_size,
        16: model_size,
        32: model_size,
        64: model_size//2,
        128: model_size//4,
        256: model_size//8,
        512: model_size//16
    },
    generator=dict(
        scalar_pose_input=False,
        type="Generator",
        unet=dict(
            enabled=True,
            residual=False
        ),
        min_fmap_resolution=4,
        use_skip=True,
        z_shape=(32, 4, 4),
        conv2d_config=dict(
            pixel_normalization=True,
            leaky_relu_nslope=.2,
            normalization="pixel_wise",
            conv=dict(
                type="conv",
                wsconv=True,
                gain=1,
            )
        ),
        residual=False,
    ),
    discriminator=dict(
        type="Discriminator",
        residual=False,
        scalar_pose_input=False,
        scalar_pose_input_imsize=32,
        min_fmap_resolution=4,
        conv_multiplier=1,
        conv2d_config=dict(
            leaky_relu_nslope=.2,
            normalization=None,
            conv=dict(
                type="conv",
                wsconv=True,
                gain=1,
            )
        ),
    )
)

trainer = dict(
    hooks=[
        dict(type="RunningAverageHook"),
        dict(
            type="CheckpointHook",
            ims_per_checkpoint=2e5
        ),
        dict(type="SigTermHook"),
        dict(
            type="ImageSaveHook",
            ims_per_save=1e5,
            n_diverse_samples=5
        ),
        dict(
            type="MetricHook",
            ims_per_log=2e5,
            lpips_batch_size=128,
            fid_batch_size=8,
            min_imsize_to_calculate=128
        ),
        dict(
            type="StatsLogger",
            num_ims_per_log=500
        )
    ],
    progressive=dict(
        transition_iters=12e5,
        minibatch_repeats=4,
        enabled=True
    ),
    batch_size_schedule={
        4: 256,
        4: 256,
        8: 256,
        16: 256,
        32: 128,
        64: 96,
        128: 32,
        256: 32
    },
    optimizer=dict(
        learning_rate=0.001,
        amp_opt_level="O1",
        lazy_regularization=True
    )
)

adversarial_loss = "WGANCriterion"
discriminator_criterions = {
    0: dict(
        type=adversarial_loss,
        fake_index=-1
    ),
    1: dict(
        type="GradientPenalty",
        lambd=10,
        mask_region_only=True,
        norm="L2",
        distance="clamp",
        lazy_reg_interval=16,
        mask_decoder_gradient=False,
        fake_index=-1
    ),
    2: dict(
        type="EpsilonPenalty",
        weight=0.001,
        fake_index=-1
    )
}
generator_criterions = {
    0: dict(
        type=adversarial_loss,
        fake_index=-1
    )
}

## DATASETS
dataset_type = "FDFDataset"
data_root = os.path.join("data", "fdf")
data_train = dict(
    dataset=dict(
        type=dataset_type,
        dirpath=os.path.join(data_root, "train"),
        percentage=1.0
    ),
    transforms=[
        dict(type="RandomFlip", flip_ratio=0.5),
        dict(type="FlattenLandmark")
    ],
    loader=dict(
        shuffle=True,
        num_workers=16,
        drop_last=True
    )
)
data_val = dict(
    dataset=dict(
        type=dataset_type,
        dirpath=os.path.join(data_root, "val"),
        percentage=0.2
    ),
    transforms=[
        dict(type="FlattenLandmark")
    ],
    loader=dict(
        shuffle=False,
        num_workers=16,
        drop_last=True
    )
)

anonymizer = dict(
    truncation_level=0,
    save_debug=False,
    batch_size=1,
    fp16_inference=False,
    jit_trace=False,
    detector_cfg=dict(
        type="BaseDetector",
        keypoint_threshold=.2,
        densepose_threshold=.3,
        simple_expand=True,
        align_faces=False,
        resize_background=True,
        face_detector_cfg=dict(
            name="RetinaNetResNet50",
            confidence_threshold=.3,
            nms_iou_threshold=.3,
            max_resolution=1080,
            fp16_inference=True,
            clip_boxes=True
        )
    )
)
