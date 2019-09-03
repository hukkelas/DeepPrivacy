import numpy as np
import torch


def to_cuda(elements):
    if torch.cuda.is_available():
        if type(elements) == tuple or type(elements) == list:
            return [x.to(get_device()) for x in elements]
        return elements.to(get_device())
    return elements


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def image_to_torch(image, cuda=True):
    if image.dtype == np.uint8:
        image = image.astype(np.float32)
    else:
        assert image.dtype == np.float32
    image = np.rollaxis(image, 2)
    image = image[None, :, :, :]
    image = torch.from_numpy(image)
    if cuda:
        image = to_cuda(image)
    return image
