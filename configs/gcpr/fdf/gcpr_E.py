
_base_config_ = "../../fdf/base.py"

models = dict(
    generator=dict(
        type="MSGGenerator",
        scalar_pose_input=True,
        conv2d_config=dict(
            conv=dict(
                type="iconv"
            )
        )
    ),
    discriminator=dict(
        residual=True,
        scalar_pose_input=True,
        scalar_pose_imsize=64
    )
)
trainer = dict(
    progressive=dict(
        enabled=False
    )
)
