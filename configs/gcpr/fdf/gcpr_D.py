
_base_config_ = "../../fdf/base.py"

models = dict(
    generator=dict(
        scalar_pose_input=True,
        conv2d_config=dict(
            conv=dict(
                type="iconv"
            )
        )
    ),
)
trainer = dict(
    progressive=dict(
        enabled=True
    )
)
