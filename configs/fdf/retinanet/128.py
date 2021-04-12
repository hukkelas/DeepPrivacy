import os

_base_config_ = "base.py"
model_url = "http://folk.ntnu.no/haakohu/checkpoints/fdf/retinanet128.ckpt"

models = dict(
    generator=dict(
        type="MSGGenerator",
        scalar_pose_input=True
    ),
    discriminator=dict(
        residual=True
    )
)

trainer = dict(
    progressive=dict(
        enabled=False,
    )
)