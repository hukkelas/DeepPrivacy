import torch 
from utils import load_checkpoint, to_cuda
from scripts.utils import init_generator, image_to_numpy, get_model_name
from train import denormalize_img, preprocess_images
from metrics import ssim, mssim
from dataloaders import load_celeba_condition
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import os

if __name__ == "__main__":
    model_name = get_model_name()
    ckpt_path = os.path.join("checkpoints", model_name)
    ckpt = load_checkpoint(ckpt_path)
    g = init_generator(ckpt)
    g.eval()
    dataloader = load_celeba_condition(10, ckpt["current_imsize"])

    mssim_all = []
    ssim_all = []
    mssim_outside_mask = []
    ssim_outside_mask = []
    mssim_inside_mask = []
    ssim_inside_mask = []
    image_channels = ckpt["image_channels"]
    transition_value = ckpt["transition_variable"]
    for orig, z in tqdm.tqdm(dataloader.validation_set_generator()):
        z = preprocess_images(z, transition_value) # 1, 64, 64

        g.train()
        d = g(z, None)
        g.eval()
        d = denormalize_img(d)
        conditions = denormalize_img(z)
        orig = to_cuda(orig)
        mssim_all.append(mssim.msssim(orig, d).data.cpu().item())
        ssim_all.append(ssim.ssim(orig, d).data.cpu().item())

        # Outside masked area
        mask = conditions.sum(dim=1,keepdim=True).repeat(1, image_channels, 1, 1) == 0
        org_masked = orig.clone()
        org_masked[mask] = 0
        images_masked = d.clone()
        images_masked[mask] = 0
        mssim_outside_mask.append(mssim.msssim(images_masked, org_masked).data.cpu().item())
        ssim_outside_mask.append(ssim.ssim(images_masked, org_masked).data.cpu().item())

        # WITHIN MASK
        mask = conditions.sum(dim=1,keepdim=True).repeat(1, image_channels, 1, 1) > 0 
        org_masked = orig.clone()
        org_masked[mask] = 0
        images_masked = d.clone()
        images_masked[mask] = 0
        mssim_inside_mask.append(mssim.msssim(images_masked, org_masked).data.cpu().item())
        ssim_inside_mask.append(ssim.ssim(images_masked, org_masked).data.cpu().item())

    print("MSSIM ALL:", np.mean(mssim_all))
    print("SSIM ALL:", np.mean(ssim_all))
    print("MSSIM OUTISDE MASK:", np.mean(mssim_outside_mask))
    print("SSIM OUTISDE MASK:", np.mean(ssim_outside_mask))
    print("MSSIM INSIDE MASK:", np.mean(mssim_inside_mask))
    print("SSIM INSIDE MASK:", np.mean(ssim_inside_mask))