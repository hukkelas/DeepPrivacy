
_base_config_ = "base.py"

model_size = 512

model_url = "http://folk.ntnu.no/haakohu/checkpoints/step_42000000.ckpt"

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
