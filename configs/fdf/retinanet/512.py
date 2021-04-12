import os

_base_config_ = "base.py"
model_url = "http://folk.ntnu.no/haakohu/checkpoints/fdf/retinanet512.ckpt"

model_size=512
models = dict(
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
        residual=True,
        scalar_pose_input=True
    ),
    discriminator=dict(
        residual=True
    )
)

trainer = dict(
    max_images_to_train=20e6,
    progressive=dict(
        enabled=False,
    ),
)