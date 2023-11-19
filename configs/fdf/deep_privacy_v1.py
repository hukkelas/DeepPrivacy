
_base_config_ = "base.py"

model_size = 512

model_url = "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/ab71cc2c-82bd-4fa7-8801-f10ff5a852246198eba4-bf7a-4e7e-9ff3-8ff6fe7f102c04c5e41b-2916-41a0-aa2b-e7a939e6f0d5"

models = dict(
    scalar_pose_input=False,
    max_imsize=128,
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
        conv2d_config=dict(
            conv=dict(
                gain=2**0.5
            )
        ),
        type="DeepPrivacyV1"),
)
trainer = dict(
    progressive=dict(
        enabled=False,
        lazy_regularization=True
    ),
    batch_size_schedule={
        128: 32,
        256: 32
    },
    optimizer=dict(
        learning_rate=0.0015
    )
)
