
_base_config_ = "../../places2.py"
models = dict(
    generator=dict(
        type="MSGGenerator",
        conv2d_config=dict(
        )
    ),
    discriminator=dict(
        residual=True,
        scalar_pose_input=False
    )
)
trainer = dict(
    progressive=dict(
        enabled=False
    ),
)