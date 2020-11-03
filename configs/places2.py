import os

_base_config_ = "defaults.py"

models = dict(
    pose_size=0,
    max_imsize=256
)

dataset_type = "Places2Dataset"
data_root = os.path.join("data", "places2")
data_train = dict(
    dataset=dict(
        type=dataset_type,
        dirpath=os.path.join(data_root, "train"),
        percentage=1.0,
        is_train=True
    ),
    transforms=[
        dict(type="RandomFlip", flip_ratio=0.5),
        dict(type="RandomCrop")
    ],
)
data_val = dict(
    dataset=dict(
        type=dataset_type,
        dirpath=os.path.join(data_root, "val"),
        percentage=.137, #5000 images out of 36500
        is_train=False
    ),
    transforms=[
        dict(type="CenterCrop")
    ],
)
