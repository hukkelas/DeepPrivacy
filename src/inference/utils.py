import numpy as np
from src.utils import to_cuda
import torch


def to_torch(image):
    image = np.rollaxis(image, 2)
    image = image[None, :, :, :]
    image = image.astype(np.float32)
    image = image / image.max()
    image = torch.from_numpy(image)
    image = to_cuda(image)
    return image

def image_to_numpy(images):
    single_image = False
    if len(images.shape) == 3:
        single_image = True
        images = images[None]
    images = images.data.detach().cpu().numpy()
    r,g,b = images[:, 0], images[:, 1], images[:, 2]
    images = np.stack((r,g,b), axis=3)
    if single_image:
        return images[0]
    return images