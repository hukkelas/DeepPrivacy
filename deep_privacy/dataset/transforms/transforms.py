import numpy as np
import albumentations
import cv2
from .build import TRANSFORM_REGISTRY


@TRANSFORM_REGISTRY.register_module
class RandomFlip:

    def __init__(self, flip_ratio=None, **kwargs):
        self.flip_ratio = flip_ratio
        if self.flip_ratio is None:
            self.flip_ratio = 0.5
        assert 0 <= self.flip_ratio <= 1

    def __call__(self, container):
        if np.random.rand() > self.flip_ratio:
            return container
        img = container["img"]
        container["img"] = np.flip(img, axis=1)
        if "landmarks" in container:
            landmarks_XY = container["landmarks"]
            landmarks_XY[:, 0] = 1 - landmarks_XY[:, 0]
            container["landmarks"] = landmarks_XY
        if "mask" in container:
            mask = container["mask"]
            mask = np.flip(mask, axis=1)
            container["mask"] = mask
        return container


@TRANSFORM_REGISTRY.register_module
class FlattenLandmark:

    def __init__(self, *args, **kwargs):
        return

    def __call__(self, container, **kwargs):
        assert "landmarks" in container,\
            f"Did not find landmarks in container. {container.keys()}"
        landmarks_XY = container["landmarks"]
        landmarks_XY = landmarks_XY.reshape(-1)
        landmarks_XY.clip(-1, 1)
        container["landmarks"] = landmarks_XY
        return container


def _resize(im, imsize):
    min_size = min(im.shape[:2])
    factor = imsize / min_size
    new_size = [int(size * factor) + 1 for size in im.shape[:2]]
    im = albumentations.augmentations.functional.resize(im, *new_size)
    return im


@TRANSFORM_REGISTRY.register_module
class RandomCrop:

    def __init__(self, imsize, **kwargs):
        self.imsize = imsize

    def __call__(self, container, **kwargs):
        im = container["img"]
        if any(size < self.imsize for size in im.shape[:2]):
            im = _resize(im, self.imsize)
        im = albumentations.augmentations.functional.random_crop(
            im, self.imsize, self.imsize, 0, 0)
        container["img"] = im
        return container


@TRANSFORM_REGISTRY.register_module
class CenterCrop:

    def __init__(self, imsize, **kwargs):
        self.imsize = imsize

    def __call__(self, container, **kwargs):
        im = container["img"]
        if any(size < self.imsize for size in im.shape[:2]):
            im = _resize(im, self.imsize)
        im = albumentations.augmentations.functional.center_crop(
            im, self.imsize, self.imsize)
        container["img"] = im
        return container


@TRANSFORM_REGISTRY.register_module
class RandomResize:

    def __init__(self, resize_ratio, min_imsize: int, max_imsize: int,
                 imsize: int, **kwargs):
        self.resize_ratio = resize_ratio
        imsize = min(min_imsize, imsize)
        self.possible_shapes = []
        while imsize <= max_imsize:
            self.possible_shapes.append(imsize)
            imsize *= 2

    def __call__(self, container, **kwargs):
        if np.random.rand() > self.resize_ratio:
            return container
        im = container["img"]
        shape = im.shape
        imsize = np.random.choice(self.possible_shapes)
        orig_imsize = im.shape[0]
        im = albumentations.augmentations.functional.resize(
            im, imsize, imsize, interpolation=cv2.INTER_LINEAR)
        im = albumentations.augmentations.functional.resize(
            im, orig_imsize, orig_imsize, interpolation=cv2.INTER_LINEAR)
        assert im.shape == shape
        container["img"] = im
        return container
