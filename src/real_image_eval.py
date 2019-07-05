import torch 
from unet_model import Generator
from utils import load_checkpoint
from train import preprocess_images, denormalize_img
from torchvision.transforms.functional import to_tensor
import matplotlib
import numpy as np
matplotlib.use("agg")
import matplotlib.pyplot as plt

import cv2
import os, glob
from dataset_tool_new import expand_bounding_box
from scripts.utils import init_generator, get_model_name, image_to_numpy
from SFD_pytorch.wider_eval_pytorch import detect_and_supress

def draw_bboxes(image, bboxes, colors):
    image = image.copy()
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        image = cv2.rectangle(image, (x0, y0), (x1, y1),colors, 1)
    return image

if __name__ == "__main__":
    model_name = get_model_name()
    ckpt_path = os.path.join("checkpoints", model_name)
    ckpt = load_checkpoint(ckpt_path)
    source_dir = os.path.join("test_examples", "real_images_test", "source")
    savedir = os.path.join("test_examples", "real_images_test", "out")
    os.makedirs(savedir, exist_ok=True)
    
    
    g = init_generator(ckpt)
    imsize = ckpt["current_imsize"]
    g.eval()
    image_paths = glob.glob(os.path.join(source_dir, "*.jpg"))
    for impath in image_paths:
        im = cv2.imread(impath) # BGR

        bounding_boxes = detect_and_supress(im)
        
        im = im[:, :, ::-1] # BGR to RGB
        new_image = im.copy()
        for idx, bbox in enumerate(bounding_boxes):
            x0, y0, x1, y1 = bbox
            width = x1 - x0 
            height = y1 - y0
            to_generate = im.copy()
            to_generate[y0:y1, x0:x1] = 0

            x0, y0, width, height = expand_bounding_box(x0, y0, width, height, 0.25, im.shape)
            print("Face shape:", (height, width))
            to_generate = to_generate[y0:y0+height, x0:x0+width]
            debug_image = to_generate.copy()
            to_generate = cv2.resize(to_generate, (imsize, imsize), interpolation=cv2.INTER_AREA)
            to_generate = to_tensor(to_generate)[None, :, :, :]
            to_generate = preprocess_images(to_generate, 1.0).cuda()
            to_generate = g(to_generate, None)
            to_generate = denormalize_img(to_generate)

            to_generate = image_to_numpy(to_generate)[0]
            to_generate = (to_generate * 255).astype("uint8")

            to_generate = cv2.resize(to_generate, (height, width))

            debug_image = np.concatenate((debug_image, to_generate), axis=1)
            debug_dir = os.path.join(savedir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, "{}.jpg".format(idx))
            plt.imsave(debug_path, debug_image)
            new_image[y0:y0+height, x0:x0+width] = to_generate
            

        imname = os.path.basename(impath)

        save_path = os.path.join(savedir, imname)
        plt.imsave(save_path, new_image)
        print("Image saved to:", save_path)


        # Save detected boxes
        save_path = imname.split(".")[0]
        save_path = "{}_detected.jpg".format(save_path)
        save_path = os.path.join(savedir, save_path)
        image = draw_bboxes(im, bounding_boxes, (255, 0, 0))
        for idx, bbox in enumerate(bounding_boxes):
            x0, y0, x1, y1 = bbox
            width = x1 - x0 
            height = y1 - y0
            x0, y0, width, height = expand_bounding_box(x0, y0, width, height, 0.25, image.shape)
            bounding_boxes[idx] = [x0, y0, x0+width, y0+height]
        image = draw_bboxes(image, bounding_boxes, (0, 0, 255))

        plt.imsave(save_path, image)
