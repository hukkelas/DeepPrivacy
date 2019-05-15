import torch 
from unet_model import Generator
import pandas as pd
from utils import load_checkpoint
from train import NetworkWrapper, denormalize_img, preprocess_images
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
#from dataset_tool_new import expand_bounding_box
from dataset_tools.utils import expand_bounding_box
from scripts.utils import init_generator, get_model_name, image_to_numpy, plot_pose, draw_bboxes, draw_keypoints
#from detectron_scripts.infer_simple_v2 import predict_keypoint
import utils



def draw_keypoints(image, keypoints, colors):
    image = image.copy()
    for keypoint in keypoints:
        X = keypoint[range(0, 14, 2)]
        Y = keypoint[range(1, 14, 2)]
        for x, y in zip(X, Y):
            cv2.circle(image, (x, y), 3, colors)
    return image

if __name__ == "__main__":
    model_name = get_model_name()
    ckpt_path = os.path.join("checkpoints", model_name)
    ckpt = load_checkpoint(ckpt_path)
    images, bounding_boxes, landmarks = load_dataset_files("data/yfcc100m128_torch", ckpt["current_imsize"])
    #images = np.zeros((128, 128, 128, 3))
    #landmarks = torch.zeros((128, 14))
    #bounding_boxes = torch.zeros((128, 4))
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
    
    orig = images[idx:idx+1].astype(np.float32) / 255
    orig_pose = landmarks[idx:idx+1]
    #print(dataloader.bounding_boxes.shape)
    x0, y0, x1, y1 = bounding_boxes[idx].numpy()
    width = x1 - x0
    height = y1 - y0
    assert orig.max() <= 1.0
    num_ims = 10
    percentages = np.linspace(0.0, 0.1, num_ims)
    shift = 2
    final_im = np.ones((imsize*2 + shift, imsize*num_ims + shift*(num_ims-1), 3), dtype=np.uint8) * 255

    for i in range(num_ims):
        p = percentages[i]
        im = orig.copy()
        pose = orig_pose.clone()
        rand = torch.rand(pose.shape) - 0.5
        rand = rand / 0.5 
        rand = rand * percentages[i]
        pose += rand
        
        cut_im = im[0].copy()
        im[0] = cut_bounding_box(im[0], [int(i) for i in [x0, y0, x1, y1]])
        
        im = torch.from_numpy(np.rollaxis(im[0],2))[None, :, :, :]
        print(im.dtype)
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
        
        cut_im = draw_keypoints(cut_im, pose*imsize, (255, 0, 0))
        # Register normal im
        final_im[:imsize, (imsize+shift)*i:(imsize+shift)*i + imsize, :] = utils.clip(cut_im * 255, 0, 255).astype(np.uint8)
        final_im[imsize+shift:, (imsize+shift)*i:(imsize+shift)*i + imsize, :] = utils.clip(im * 255, 0, 255).astype(np.uint8)
    plt.imsave(os.path.join(savedir, "pose_sensitivity.png"), final_im)
    print("Results saved to:", savedir)
