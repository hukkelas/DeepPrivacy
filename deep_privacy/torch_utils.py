import numpy as np
import torch
import math


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def image_to_numpy(images, to_uint8=False, denormalize=False):
    single_image = False
    if len(images.shape) == 3:
        single_image = True
        images = images[None]
    if denormalize:
        images = denormalize_img(images)
    images = images.detach().cpu().numpy()
    images = np.moveaxis(images, 1, -1)
    if to_uint8:
        images = (images * 255).astype(np.uint8)
    if single_image:
        return images[0]
    return images


def denormalize_img(image):
    image = (image + 1) / 2
    image = torch.clamp(image.float(), 0, 1)
    return image


def number_of_parameters(module: torch.nn.Module):
    count = 0
    for p in module.parameters():
        count += np.prod(p.shape)
    return count


def image_to_torch(image, cuda=True, normalize_img=False):
    single_image = len(image.shape) == 3
    if image.dtype == np.uint8:
        image = image.astype(np.float32)
        image /= 255
    else:
        assert image.dtype == np.float32
    if single_image:
        image = np.rollaxis(image, 2)
        image = image[None, :, :, :]
    else:
        image = np.moveaxis(image, -1, 1)
    image = torch.from_numpy(image).contiguous()
    if cuda:
        image = to_cuda(image)
    assert image.min() >= 0.0 and image.max() <= 1.0
    if normalize_img:
        image = image * 2 - 1
    return image


def mask_to_torch(mask: np.ndarray, cuda=True):
    assert mask.max() <= 1 and mask.min() >= 0
    mask = mask.squeeze()
    single_mask = len(mask.shape) == 2
    if single_mask:
        mask = mask[None]
    mask = mask[:, None, :, :]
    mask = torch.from_numpy(mask)
    if cuda:
        mask = to_cuda(mask)
    return mask


def _to_cuda(element):
    if isinstance(element, torch.nn.Module):
        return element.cuda()
    return element.cuda(non_blocking=True)


def to_cuda(elements):
    if torch.cuda.is_available():
        if isinstance(elements, tuple) or isinstance(elements, list):
            return [_to_cuda(x) for x in elements]
        return _to_cuda(elements)
    return elements


def isinf(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("The argument is not a tensor", str(tensor))
    return (tensor.abs() == math.inf).any()


def isnan(tensor):
    r"""Returns a new tensor with boolean elements representing if each element
    is `NaN` or not.
    Arguments:
        tensor (Tensor): A tensor to check
    Returns:
        Tensor: A ``torch.ByteTensor`` containing a 1 at each location of `NaN`
        elements.
    Example::
        >>> torch.isnan(torch.tensor([1, float('nan'), 2]))
        tensor([ 0,  1,  0], dtype=torch.uint8)
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("The argument is not a tensor", str(tensor))
    return (tensor != tensor).any()


def finiteCheck(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    for p in parameters:
        if isinf(p.grad.data):
            return False
        if isnan(p.grad.data):
            return False
    return True


def keypoint_to_numpy(keypoint):
    keypoint = keypoint.cpu()
    return keypoint.view(-1, 2).numpy()
