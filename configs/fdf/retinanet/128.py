import os

_base_config_ = "base.py"
model_url = "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/89275817-b6b9-4523-9b7a-885d6164d6f8c23269b9-37a6-4dc2-9cbe-048a60b0622b558694fb-e1f6-403c-a6db-13fb4019f59e"

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