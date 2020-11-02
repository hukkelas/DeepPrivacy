import torchvision
from deep_privacy.utils import Registry, build_from_cfg

TRANSFORM_REGISTRY = Registry("TRANSFORM")


def build_transforms(transforms, imsize):
    transforms = [
        build_from_cfg(t, TRANSFORM_REGISTRY, imsize=imsize)
        for t in transforms]
    return torchvision.transforms.Compose(transforms)
