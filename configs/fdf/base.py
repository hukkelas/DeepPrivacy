import os

_base_config_ = "../defaults.py"

dataset_type = "FDFDataset"
data_root = os.path.join("data", "fdf")
data_train = dict(
    dataset=dict(
        type=dataset_type,
        dirpath=os.path.join(data_root, "train"),
        percentage=1.0
    ),
    transforms=[
        dict(type="RandomFlip", flip_ratio=0.5),
        dict(type="FlattenLandmark")
    ],
)
data_val = dict(
    dataset=dict(
        type=dataset_type,
        dirpath=os.path.join(data_root, "val"),
        percentage=.2
    ),
    transforms=[
        dict(type="FlattenLandmark")
    ],
)

landmarks = [
    "Nose",
    "Left Eye",
    "Right Eye",
    "Left Ear",
    "Right Ear",
    "Left Shoulder",
    "Right Shoulder"
]
anonymizer = dict(
    detector_cfg=dict(
        type="RCNNDetector",
        simple_expand=False,
        rcnn_batch_size=8,
        face_detector_cfg=dict(
            name="RetinaNetResNet50",
        )
    )
)