
_base_config_ = "base.py"
model_size = 512
model_url = "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/0940803d-1a2c-4b54-9c1d-0d3aeb80c5a8970f7dd4-bde6-4120-9d3e-5fad937eef2b7b0544e2-8c00-43c2-9193-f56679740872"
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
