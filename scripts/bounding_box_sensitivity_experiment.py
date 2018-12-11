import torch 
from unet_model import Generator
import pandas as pd
from utils import load_checkpoint
from train import DataParallellWrapper, normalize_img, adjust_dynamic_range, preprocess_images
import torchvision
from dataloaders import load_mnist, load_celeba_condition
from torchvision.transforms.functional import to_tensor, to_pil_image
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision.transforms.functional import to_tensor
import os
from test import expand_bounding_box
from scripts.utils import init_generator, get_model_name
if __name__ == "__main__":
    model_name = get_model_name()
    ckpt_path = os.path.join("checkpoints", model_name)
    ckpt = load_checkpoint(ckpt_path)

    savedir = os.path.join("test_examples", "bounding_box_test")
    os.makedirs(savedir, exist_ok=True)
    CELEBA_DATA_DIR = os.path.join("data","celeba","img_celeba")
    CELEBA_BBOX_FILE = os.path.join("data","celeba","list_bbox_celeba.txt")
    ""
    g = init_generator(ckpt)
    imsize = ckpt["current_imsize"]
    g.eval()

    # Read bounding box annotations
    bbox_df = pd.read_csv(CELEBA_BBOX_FILE, skiprows=[0], delim_whitespace=True)
    idx = 6
    imname = bbox_df.iloc[idx].image_id
    d = bbox_df.iloc[idx]
    x0, y0, width, height = d.x_1, d.y_1, d.width, d.height 
    impath = os.path.join(CELEBA_DATA_DIR, imname)

    orig = plt.imread(impath).copy()
    x1 = x0 + width
    y1 = y0 + height
    x0_, y0_, width_, height_ = expand_bounding_box(x0, y0, width, height, 0.25, orig.shape)

    x1_ = x0_ + width_
    y1_ = y0_ + height_

    percentages = np.linspace(-0.3, 0.3, 100)
    for i in range(100):
        p = percentages[i]
        x0_altered = x0 - int(p*width)
        x1_altered = x1 + int(p*width)
        y0_altered = max(y0 - int(p*height),0)
        y1_altered = min(y1 + int(p*height), y1_)
        im = orig.copy()
        im[y0_altered:y1_altered, x0_altered:x1_altered] = 0
        im = im[y0_:y1_, x0_:x1_]
        
        
        im = cv2.resize(im, (imsize, imsize), interpolation=cv2.INTER_AREA)
        plt.imsave("{}/{:.3f}in.jpg".format(savedir, p), im)
        
        im = to_tensor(im)[None, :, :, :]
        
        im = preprocess_images(im, 1.0).cuda()

        im = g(im, None)
        im = normalize_img(im)
        torchvision.utils.save_image(im.data, "{}/{:.3f}out.jpg".format(savedir, p))
    print("Results saved to:", savedir)
