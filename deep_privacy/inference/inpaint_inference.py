import numpy as np
import torch
import tqdm
from deep_privacy import torch_utils
from .infer import truncated_z


def inpaint_images(
        images: np.ndarray, masks: np.ndarray,
        generator):
    z = None
    fakes = torch.zeros(
        (images.shape[0], images.shape[-1], images.shape[1], images.shape[2]),
        dtype=torch.float32)
    masks = pre_process_masks(masks)
    inputs = [im * mask for im, mask in zip(images, masks)]
    images = [
        torch_utils.image_to_torch(im, cuda=False, normalize_img=True)
        for im in images]
    masks = [torch_utils.mask_to_torch(mask, cuda=False) for mask in masks]
    with torch.no_grad():
        for idx, (im, mask) in enumerate(
                tqdm.tqdm(zip(images, masks), total=len(images))):
            im = torch_utils.to_cuda(im)
            mask = torch_utils.to_cuda(mask)
            assert im.shape[0] == mask.shape[0]
            assert im.shape[2:] == mask.shape[2:],\
                f"im shape: {im.shape}, mask shape: {mask.shape}"
            z = truncated_z(im, generator.z_shape, 0)
            condition = mask * im
            fake = generator(condition, mask, z)
            fakes[idx:(idx + 1)] = fake.cpu()
    fakes = torch_utils.image_to_numpy(fakes, denormalize=True) * 255
    return fakes, inputs


def pre_process_masks(masks: np.ndarray):
    if masks.shape[-1] == 4:
        masks = masks[:, :, :3]
    if masks.shape[-1] == 3:
        masks = masks.mean(axis=-1, keepdims=True)
    masks = (masks > 0).astype(np.float32)
    masks = np.ascontiguousarray(masks)
    return masks
