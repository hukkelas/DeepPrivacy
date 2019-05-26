import torch 
from unet_model import Generator
import pandas as pd
from utils import load_checkpoint
from train import  denormalize_img, preprocess_images
import torchvision
from dataloaders_v2 import load_dataset_files, cut_bounding_box
from torchvision.transforms.functional import to_tensor, to_pil_image
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision.transforms.functional import to_tensor
import os
from scripts.utils import init_generator, get_model_name, image_to_numpy, plot_pose
#from detectron_scripts.infer_simple_v2 import predict_keypoint
import utils

if __name__ == "__main__":
    model_name = get_model_name()
    ckpt_path = os.path.join("checkpoints", model_name)
    ckpt = load_checkpoint(ckpt_path)

    images, bounding_boxes, landmarks = load_dataset_files("data/yfcc100m128_torch", ckpt["current_imsize"])

    savedir = os.path.join("test_examples", "z_noise_test")
    os.makedirs(savedir, exist_ok=True)
    g = init_generator(ckpt)
    imsize = ckpt["current_imsize"]
    g.eval()

    # Read bounding box annotations
    idx = -5

    orig = images[idx:idx+1].astype(np.float32) / 255
    pose = landmarks[idx:idx+1]
    x0, y0, x1, y1 = bounding_boxes[idx].numpy()
    width = x1 - x0
    height = y1 - y0
    assert orig.max() <= 1.0

    to_save = orig.squeeze()
    num_iterations = 20
    max_levels = np.linspace(0.2, 2.0, num_iterations)
    final_images = []
    for i in range(num_iterations):
        im = orig.copy()
        im[0] = cut_bounding_box(im[0], [int(x) for x in [x0, y0, x1, y1]])
        
        im = torch.from_numpy(np.rollaxis(im[0], 2))[None, :, :, :]
        assert list(im.shape) == [1, 3, imsize, imsize], "Shape was:{}".format(im.shape)
        im = preprocess_images(im, 1.0).cuda()
        im = g(im, pose)
        im = denormalize_img(im)
        
        im = image_to_numpy(im.squeeze())
        to_save = np.concatenate((to_save, im), axis=1)
        final_images.append(im)
    for j in range(num_iterations-1):
        diff = abs(final_images[j] - final_images[j+1])
        print("Diff {}: {}".format(j, diff.mean()))
    savepath = os.path.join(savedir, "result_image_trunc.jpg")
    plt.imsave(savepath, to_save)

    print("Results saved to:", savedir)
    to_save = np.zeros((imsize*num_iterations, imsize*num_iterations, 3))
    savedir = os.path.join(savedir, "diff_results")
    os.makedirs(savedir, exist_ok=True)
    for r in range(num_iterations):
        for c in range(num_iterations):
            im1 = final_images[r]
            im2 = final_images[c]
            diff = abs(im2 - im1)
            print(r,c, diff.max())
            if diff.max() == 0:
                diff[0,0, 0] = 1
            diff = diff / diff.max()
            to_save = np.column_stack((im1, im2, diff))
            savepath = os.path.join(savedir, "res_{}_{}.jpg".format(r, c))
            plt.imsave(savepath, to_save)

