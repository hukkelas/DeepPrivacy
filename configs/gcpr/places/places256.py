
_base_config_ = "../../places2.py"
model_size = 256
models = dict(
    scalar_pose_input=False,
    max_imsize=256,
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
    pose_size=0,
    generator=dict(
        type="MSGGenerator",
        conv2d_config=dict(
            conv=dict(
                type="iconv"
            )
        )
    ),
    discriminator=dict(
        residual=True,
        scalar_pose_input=False
    )
)
trainer = dict(
    max_images_to_train=20e6,
    progressive=dict(
        enabled=False
    ),
    batch_size_schedule={
        256: 32
    },
    optimizer=dict(
        learning_rate=0.0015
    )
)
