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

    savedir = os.path.join("test_examples", "bounding_box_test")
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(os.path.join(savedir, "out"), exist_ok=True)
    os.makedirs(os.path.join(savedir, "in"), exist_ok=True)
    os.makedirs(os.path.join(savedir, "debug"), exist_ok=True)
    g = init_generator(ckpt)
    imsize = ckpt["current_imsize"]
    g.eval()

    # Read bounding box annotations
    idx = -5
    
    orig = dataloader.images[idx:idx+1]
    pose = dataloader.landmarks[idx:idx+1]
    print(dataloader.bounding_boxes.shape)
    x0, y0, x1, y1 = dataloader.bounding_boxes[idx].numpy()
    width = x1 - x0
    height = y1 - y0
    assert orig.max() <= 1.0
    
    percentages = np.linspace(-0.3, 0.3, 100)
    for i in range(100):
        p = percentages[i]
        x0_altered = x0 - int(p*width)
        x1_altered = x1 + int(p*width)
        y0_altered = max(y0 - int(p*height), 0)
        y1_altered = min(y1 + int(p*height), imsize)
        im = orig.clone()
        
        to_replace = im[:, :, y0_altered:y1_altered, x0_altered:x1_altered]
        m = (to_replace / im.max()).mean()
        s = (to_replace / im.max()).std()

        norm = utils.truncated_normal(m, s, to_replace.shape)
        to_replace[:, :, :, :] = norm
        assert list(im.shape) == [1, 3, imsize, imsize], "Shape was:{}".format(im.shape)

        torchvision.utils.save_image(im.data, "{}/in/{:.3f}.jpg".format(savedir, percentages[i]))
        to_debug = image_to_numpy(im.squeeze())
        im = preprocess_images(im, 1.0).cuda()
        #pose = torch.zeros((1, 14))
        im = g(im, pose)
        im = denormalize_img(im)
        
        torchvision.utils.save_image(im.data, "{}/out/{:.3f}.jpg".format(savedir, percentages[i]))

        im = image_to_numpy(im.squeeze())
        to_debug = np.concatenate((to_debug, im), axis=1)
        plt.figure(figsize=(24, 12))
        plt.imshow(to_debug)
        plot_pose(pose.numpy(), imsize, 0)
        plot_pose(pose.numpy(), imsize, imsize)
        plt.savefig(os.path.join(savedir, "debug", "{:.3f}.jpg".format(percentages[i])))
        plt.close()

    print("Results saved to:", savedir)
