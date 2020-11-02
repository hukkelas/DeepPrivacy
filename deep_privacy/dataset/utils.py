import numpy as np
import torch
from PIL import Image, ImageOps
from deep_privacy.modeling.models.utils import get_transition_value


def read_image(filepath, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR"

    Returns:
        image (np.ndarray): an HWC image in the given format.
    """
    image = Image.open(filepath)

    # capture and ignore this bug:
    # https://github.com/python-pillow/Pillow/issues/3973
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format == "BGR":
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    if format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)
    return image


def fast_collate(batch):
    has_landmark = "landmarks" in batch[0]
    imshape = batch[0]["img"].shape[:2]
    images = torch.zeros(
        (len(batch), 3, *imshape), dtype=torch.uint8)
    masks = torch.zeros(
        (len(batch), 1, *imshape), dtype=torch.bool)
    if has_landmark:
        landmark = batch[0]["landmarks"]
        landmarks = torch.zeros(
            (len(batch), *landmark.shape), dtype=torch.float32
        )
    for i, sample in enumerate(batch):
        img = sample["img"]
        img = np.rollaxis(img, 2)
        images[i] += torch.from_numpy(img.copy())

        mask = torch.from_numpy(sample["mask"].copy())
        masks[i, 0] += mask
        if has_landmark:
            landmark = sample["landmarks"]
            landmarks[i] += torch.from_numpy(landmark)
    res = {"img": images, "mask": masks}
    if has_landmark:
        res["landmarks"] = landmarks
    return res


class DataPrefetcher:

    def __init__(self,
                 loader: torch.utils.data.DataLoader,
                 infinite_loader: bool):
        self.original_loader = loader
        self.stream = torch.cuda.Stream()
        self.loader = iter(self.original_loader)
        self.infinite_loader = infinite_loader

    def _preload(self):
        try:
            self.container = next(self.loader)
            self.stop_iteration = False
        except StopIteration:
            if self.infinite_loader:
                self.loader = iter(self.original_loader)
                return self._preload()
            self.stop_iteration = True
            return
        with torch.cuda.stream(self.stream):
            for key, item in self.container.items():
                self.container[key] = item.cuda(non_blocking=True).float()
            self.container["img"] = self.container["img"] * 2 / 255 - 1
            self.container["condition"] = self.container["img"] * self.container["mask"]

    def __len__(self):
        return len(self.original_loader)

    def __next__(self):
        return self.next()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.stop_iteration:
            raise StopIteration
        container = self.container
        self._preload()
        return container

    def __iter__(self):
        self.loader = iter(self.original_loader)
        self._preload()
        return self

    def num_images(self):
        assert self.original_loader.drop_last
        return len(self.original_loader) * self.original_loader.batch_size

    @property
    def batch_size(self):
        return self.original_loader.batch_size


@torch.no_grad()
def interpolate_mask(mask, transition_variable):
    y = torch.nn.functional.avg_pool2d(mask, 2)
    y = torch.nn.functional.interpolate(y, scale_factor=2, mode="nearest")
    mask = get_transition_value(y, mask, transition_variable)
    mask = (mask >= 0.5).float()
    return mask


@torch.no_grad()
def interpolate_image(images, transition_variable):
    assert images.max() <= 1
    y = torch.nn.functional.avg_pool2d(images * 255) // 1
    y = torch.nn.functional.interpolate(y, scale_factor=2)
    images = get_transition_value(y, images, transition_variable)
    return images


@torch.no_grad()
def interpolate_landmarks(landmarks, transition_variable, imsize):
    prev_landmarks = (landmarks * imsize / 2) // 1
    prev_landmarks = prev_landmarks / (imsize * 2)
    cur_landmarks = (landmarks * imsize) // 1
    cur_landmarks = cur_landmarks / imsize
    return get_transition_value(
        prev_landmarks, cur_landmarks, transition_variable)


def progressive_decorator(func, get_transition_value):
    def decorator(*args, **kwargs):
        batch = func(*args, **kwargs)
        img = batch["img"]
        batch["img"] = interpolate_image(img, get_transition_value())
        batch["mask"] = interpolate_mask(
            batch["mask"],
            get_transition_value()
        )
        if "landmarks" in batch:
            imsize = img.shape[-1]
            landmarks = batch["landmarks"]
            batch["landmarks"] = interpolate_landmarks(
                landmarks, get_transition_value(),
                imsize
            )
        return batch
    return decorator
