import os

_base_config_ = "base.py"
model_url = "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/e5b649ec-d21c-4169-9c64-a80007c975a428bad84f-0460-4474-9667-3960fb27de75696d6db1-c8f6-4bcd-845f-1b09ca191883"

model_size=256
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
    progressive=dict(
        enabled=False,
    ),
)