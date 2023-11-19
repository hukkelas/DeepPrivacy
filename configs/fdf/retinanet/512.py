import os

_base_config_ = "base.py"
model_url = "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/c24e2f70-391e-491b-b540-ddd439088d536743f3cb-5e22-44b0-8a68-c41831a5c89515d370dd-40fe-40d2-b959-e500f2cefc4d"

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