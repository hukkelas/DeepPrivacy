import torch 
from unet_model import Generator
import pandas as pd
from utils import load_checkpoint
from train import DataParallellWrapper, denormalize_img, adjust_dynamic_range, preprocess_images
import torchvision
from dataloaders import load_celeba_condition
from torchvision.transforms.functional import to_tensor, to_pil_image
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision.transforms.functional import to_tensor
import os
from dataset_tool_new import expand_bounding_box
from scripts.utils import init_generator, get_model_name, image_to_numpy, plot_pose
#from detectron_scripts.infer_simple_v2 import predict_keypoint
import utils

if __name__ == "__main__":
    model_name = get_model_name()
    ckpt_path = os.path.join("checkpoints", model_name)
    ckpt = load_checkpoint(ckpt_path)

    dataloader = load_celeba_condition(1, ckpt["current_imsize"])

    savedir = os.path.join("test_examples", "z_noise_test")
    os.makedirs(savedir, exist_ok=True)
    g = init_generator(ckpt)
    imsize = ckpt["current_imsize"]
    g.eval()

    # Read bounding box annotations
    idx = -5

    orig = dataloader.images[idx:idx+1]
    pose = dataloader.landmarks[idx:idx+1]
    x0, y0, x1, y1 = dataloader.bounding_boxes[idx].numpy()
    width = x1 - x0
    height = y1 - y0
    assert orig.max() <= 1.0

    to_save = image_to_numpy(orig.squeeze())
    num_iterations = 20
    max_levels = np.linspace(0.2, 2.0, num_iterations)
    for i in range(num_iterations):
        im = orig.clone()
        to_replace = im[:, :, y0:y1, x0:x1]
        m = (to_replace / im.max()).mean()
        s = (to_replace / im.max()).std()
        max_level = max_levels[i]
        if max_level == 0:
            norm = torch.zeros_like(to_replace)
        else:
            norm = utils.truncated_normal_old(m, s, to_replace.shape)
        print(norm.max(), norm.min())
        to_replace[:, :, :, :] = norm
        assert list(im.shape) == [1, 3, imsize, imsize], "Shape was:{}".format(im.shape)

        im = preprocess_images(im, 1.0).cuda()
        im = g(im, pose)
        im = denormalize_img(im)
        
        im = image_to_numpy(im.squeeze())
        to_save = np.concatenate((to_save, im), axis=1)
    savepath = os.path.join(savedir, "result_image_trunc.jpg")
    plt.imsave(savepath, to_save)

    print("Results saved to:", savedir)
