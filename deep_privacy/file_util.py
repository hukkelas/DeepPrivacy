import pathlib
import typing
import numpy as np
from . import utils


def find_all_files(directory: pathlib.Path,
                   suffixes=["png", "jpg", "jpeg"]
                   ) -> typing.List[pathlib.Path]:
    image_paths = []
    for suffix in suffixes:
        image_paths.extend(
            directory.glob(f"*.{suffix}")
        )
    image_paths.sort()
    return image_paths


def find_matching_files(new_directory: pathlib.Path,
                        filepaths: typing.List[pathlib.Path]
                        ) -> typing.List[pathlib.Path]:
    new_files = []
    for impath in filepaths:
        mpath = new_directory.joinpath(impath.name)
        assert mpath.is_file(), f"Did not find path: {mpath}"
        new_files.append(mpath)
    assert len(new_files) == len(filepaths)
    return new_files


def read_images(filepaths: typing.List[pathlib.Path]) -> np.ndarray:
    im0 = utils.read_im(filepaths[0])
    images = np.zeros((len(filepaths), *im0.shape), dtype=im0.dtype)
    for idx, impath in enumerate(filepaths):
        images[idx] = utils.read_im(impath)
    return images


def _is_same_shape(images: typing.List[np.ndarray]):
    shape1 = images[0].shape
    for im in images:
        if im.shape != shape1:
            return False
    return True


def read_mask_images(image_paths: typing.List[pathlib.Path],
                     mask_paths: typing.List[pathlib.Path],
                     imsize: int):
    images = [utils.read_im(impath, imsize) for impath in image_paths]
    masks = [utils.read_im(impath, imsize) for impath in mask_paths]
    if _is_same_shape(images):
        images = np.concatenate([im[None] for im in images], axis=0)
        masks = np.concatenate([im[None] for im in masks], axis=0)
    return images, masks
