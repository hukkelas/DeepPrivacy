
_base_config_ = "base.py"
model_size = 512
model_url = "http://folk.ntnu.no/haakohu/checkpoints/fdf128_model512.ckpt"
models = dict(
    scalar_pose_input=True,
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
        type="MSGGenerator"),
    discriminator=dict(
        residual=True,
        scalar_pose_input=False
    )
)
trainer = dict(
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
