import torch
import torchvision
from setuptools import setup, find_packages
torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 7], "Requires PyTorch >= 1.7"
torchvision_ver = [int(x) for x in torchvision.__version__.split(".")[:2]]
assert torchvision_ver >= [0, 6], "Requires torchvision >= 0.6"

setup(
    name='DeepPrivacy',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "cython",
        "scikit-learn>=0.2",
        "matplotlib",
        "tqdm",
        "tflib",
        "autopep8",
        "moviepy",
        "tensorboard",
        "opencv-python",
        "requests",
        "pyyaml",
        "scikit-image",
        "addict",
        "albumentations",
        "face_detection>=0.2.0"
    ]
)