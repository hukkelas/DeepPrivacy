import torch 
from unet_model import Generator
from utils import load_checkpoint
from train import preprocess_images, normalize_img
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
from detectron_scripts.infer_simple_v2 import predict_keypoint
def draw_bboxes(image, bboxes, colors):
    image = image.copy()
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        image = cv2.rectangle(image, (x0, y0), (x1, y1),colors, 1)
    return image

def is_keypoint_within_bbox(x0, y0, width, height, keypoint):
    keypoint = keypoint[:, :3]
    kp_X = keypoint[0, :]
    kp_Y = keypoint[1, :]
    within_X = np.all(kp_X >= x0) and np.all(kp_X <= x0 + width)
    within_Y = np.all(kp_Y >= y0) and np.all(kp_Y <= y0 + height)
    return within_X and within_Y


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
        #if not "reals" in impath: continue
        im = cv2.imread(impath) # BGR
        #im = im[:128, :128]
        keypoints = predict_keypoint(impath)
        keypoints = keypoints[:, :2] # Only want nose, left eye, right eye
        bounding_boxes = detect_and_supress(im)
        orig_keypoints = keypoints.copy()
        im = im[:, :, ::-1] # BGR to RGB
        new_image = im.copy()
        print("IMAGE:", impath)
        for idx, bbox in enumerate(bounding_boxes):
            
            x0, y0, x1, y1 = bbox
            width = x1 - x0 
            height = y1 - y0
            to_generate = im.copy()
            orig = im.copy()
            to_generate[y0:y1, x0:x1] = 0

            x0, y0, width, height = expand_bounding_box(x0, y0, width, height, 0.25, im.shape)
            final_keypoint = None

            for j, keypoint in enumerate(keypoints):

                if is_keypoint_within_bbox(x0, y0, width, height, keypoint):
                    final_keypoint = keypoint
                    keypoints = np.delete(keypoints, j, axis=0)
                    break
            if final_keypoint is None:
                print("Skipping bounding box.")
                continue 
                
            
            print("Face shape:", (height, width))
            to_generate = to_generate[y0:y0+height, x0:x0+width]
            orig = orig[y0:y0+height, x0:x0+width]
            final_keypoint[0, :] -= x0
            final_keypoint[1, :] -= y0

            print(final_keypoint.shape)
            #final_keypoint = final_keypoint[:]
            
            final_keypoint = final_keypoint / width
            #final_keypoint = final_keypoint / imsize # Normalize to [0, 1]
            print("pre:", final_keypoint)
            final_keypoint = np.array([final_keypoint[j, i] for i in range(final_keypoint.shape[1]) for j in range(2)])
    
            final_keypoint = torch.from_numpy(final_keypoint).view(1, -1)
            print("aft:", final_keypoint)
            to_generate = cv2.resize(to_generate, (imsize, imsize), interpolation=cv2.INTER_AREA)
            debug_image = to_generate.copy()
            to_generate = to_tensor(to_generate)[None, :, :, :]
            to_generate = preprocess_images(to_generate, 1.0).cuda()
            print(to_generate.shape)
            to_generate = g(to_generate, None, final_keypoint) #  leye_x, leye_y,reye_x, reye_y, nose_x, nose_y,
            to_generate = normalize_img(to_generate)

            to_generate = image_to_numpy(to_generate)[0]
            to_generate = (to_generate * 255).astype("uint8")
            orig = cv2.resize(orig, (imsize, imsize), interpolation=cv2.INTER_AREA)
            debug_image = np.concatenate((orig, debug_image, to_generate), axis=1)

            to_generate = cv2.resize(to_generate, (height, width))

            final_keypoint = final_keypoint.numpy().squeeze() * imsize
            print("final:", final_keypoint)

            plt.clf()
            plt.imshow(debug_image)
            
            X = final_keypoint[range(0, len(final_keypoint), 2)]
            Y = final_keypoint[range(1, len(final_keypoint), 2)]            

            plt.plot(X,Y, "o")
            plt.plot(X+2*imsize, Y, "o")

            debug_dir = os.path.join(savedir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, "{}_{}.jpg".format(os.path.basename(impath).split(".")[0], idx))
            plt.savefig(debug_path)
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
        for keypoint in orig_keypoints:
            X = keypoint[0, :]
            Y = keypoint[1, :]
            for x,y in zip(X, Y):
                cv2.circle(image, (x,y), 5, (255, 0, 0))
        image = draw_bboxes(image, bounding_boxes, (0, 0, 255))

        plt.imsave(save_path, image)
