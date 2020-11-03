
_base_config_ = "../../places2.py"
models = dict(
    scalar_pose_input=False,
    max_imsize=256,
    pose_size=0,
    generator=dict(
        conv2d_config=dict(
            conv=dict(
                type="iconv"
            )
        )
    ),
)