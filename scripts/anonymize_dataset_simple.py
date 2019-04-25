import os
import numpy as np
import cv2
import torch
from optparse import OptionParser
from utils import load_checkpoint
from scripts.utils import init_generator, image_to_numpy
from detectron_scripts.infer_simple_v2 import predict_keypoint
from dataset_tools.utils import expand_bounding_box, is_keypoint_within_bbox
from dataloaders_v2 import cut_bounding_box
from train import preprocess_images, denormalize_img
from torchvision.transforms.functional import to_tensor
from scripts.utils import draw_bboxes, draw_keypoints
from scripts.anonymize_dataset import anonymize_image, get_all_bounding_boxes, get_detectron_keypoint
import tqdm
parser = OptionParser()
parser.add_option("--source-dir", dest="source_dir",
                  help="Set the source directory of dataset")
parser.add_option("--target-dir", dest="target_dir",
                  help="Set the directory to save anonymized dataset")
parser.add_option("--model-name", dest="model_name",
                  help="Set the model name to use for anonymization")

opts, _ = parser.parse_args()

num_ignored = 0
total = 0
def anonymize_single_bbox(image, keypoints, bbox, generator, imsize):
    x0, y0, x1, y1 = bbox
    try:
        x0_, y0_, w_, h_ = expand_bounding_box(*bbox, 0.35, image.shape)
    except AssertionError:
        #print("Could not process image")
        return None 
    x1_, y1_ = x0_ + w_, y0_ + h_
    to_generate = image[y0_:y1_, x0_:x1_].copy()
    mean, std = image.mean(), image.std()

    # Shift and scale original bounding box 
    x0, x1 = x0 - x0_, x1 - x0_
    y0, y1 = y0 - y0_, y1 - y0_

    #to_generate[y0:y1, x0:x1, :] = np.random.normal(mean, std, size=(y1-y0, x1 - x0, 3))
    # Match keypoint
    final_keypoint = None 
    for j, keypoint in enumerate(keypoints):
        if is_keypoint_within_bbox(*bbox, keypoint):
            final_keypoint = keypoint
            keypoints = np.delete(keypoints, j, axis=0)
            break
    if final_keypoint is None:
        global num_ignored
        num_ignored += 1 
        return None
    orig_keypoint = final_keypoint.copy()
    # Shift and scale original keypoints
    return to_generate, orig_keypoint

def anonymize_image(image, keypoints, bounding_boxes, generator, imsize, anonymize_single_bbox):
    image = image.copy()
    keypoints = keypoints.copy()
    replaced_mask = np.ones_like(image).astype("bool")
    replaced_bboxes = []
    replaced_keypoints = []
    global total 
    for bbox in bounding_boxes:

        total += 1
        try:
            x0_, y0_, w_, h_ = expand_bounding_box(*bbox, 0.35, image.shape)
        except AssertionError as e:
            print("Could not process image, bbox error", e)
            global num_ignored
            num_ignored += 1
            continue 
        result = anonymize_single_bbox(image, keypoints, bbox, generator, imsize)
        if result  is None:
            #print("Could not process image")
            continue
        generated_face, final_keypoint = result 
        replaced_keypoints.append(final_keypoint)
        replaced_bboxes.append(bbox)
        # Mask image to not replace already generated parts 
        replaced_mask_cut = replaced_mask[y0_:y0_+h_, x0_:x0_+w_]
        to_replace = image[y0_:y0_+h_, x0_:x0_+w_]
        to_replace[replaced_mask_cut] = generated_face[replaced_mask_cut]
        image[y0_:y0_+h_, x0_:x0_+w_] = to_replace
    #image = draw_bboxes(image, replaced_bboxes, (0, 0, 255))
    #image = draw_keypoints(image, replaced_keypoints, (0, 0, 255))
    return image

# Initialize model
ckpt_path = os.path.join("checkpoints", opts.model_name)
ckpt = load_checkpoint(ckpt_path)

imsize = ckpt["current_imsize"]
bounding_boxes = get_all_bounding_boxes(opts)
generator = None
detectron_predictions = {}

for filepath in tqdm.tqdm(bounding_boxes.keys()):

    #if  "16_Award_Ceremony_Awards_Ceremony_16_546.jpg" not in filepath: continue
    total_filepath = os.path.join(opts.source_dir, filepath)
    im = cv2.imread(total_filepath).copy() # BGR
    im = im[:, :, ::-1] # BGR to RGB
    im_bounding_boxes = bounding_boxes[filepath]
    image_keypoints = get_detectron_keypoint(total_filepath, 0)#

    if image_keypoints is None or image_keypoints.tolist() is None  or  len(image_keypoints) == 0:
        new_filepath = os.path.join(opts.target_dir, filepath)
        cv2.imwrite(new_filepath, im[:, :, ::-1])
        total += len(im_bounding_boxes)
        num_ignored += len(im_bounding_boxes)
        continue
    image_keypoints = image_keypoints[:, :2, :ckpt["pose_size"]]
    anonymized_image = anonymize_image(im,
                                       image_keypoints,
                                       im_bounding_boxes,
                                       generator,
                                       imsize,
                                       anonymize_single_bbox)
    new_filepath = os.path.join(opts.target_dir, filepath)
    filedir = os.path.dirname(new_filepath)
    os.makedirs(filedir, exist_ok=True)
    print("Saving to:", new_filepath)
    #cv2.imwrite(new_filepath, anonymized_image[:, :, ::-1])
print(num_ignored / total)