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
import tqdm

image_idx = 0

def anonymize_single_bbox(image, keypoints, bbox, generator, imsize, resize=True):
    x0, y0, x1, y1 = bbox
    try:
        x0_, y0_, w_, h_ = expand_bounding_box(*bbox, 0.35, image.shape)
    except AssertionError as e:
        print("Could not process image", e)
        return None 
    x1_, y1_ = x0_ + w_, y0_ + h_
    to_generate = image[y0_:y1_, x0_:x1_].copy()

    # Shift and scale original bounding box 
    x0, x1 = x0 - x0_, x1 - x0_
    y0, y1 = y0 - y0_, y1 - y0_
    x0, y0, x1, y1 = [int(k/w_ * imsize) for k in [x0, y0, x1, y1]]
    
    # Resize to expected image size for generator
    to_generate = cv2.resize(to_generate, (imsize, imsize), interpolation=cv2.INTER_AREA)
    to_generate = cut_bounding_box(to_generate, [x0, y0, x1, y1])
    to_save = to_generate.copy()
    to_generate = to_tensor(to_generate)[None, :, :, :]

    # Match keypoint
    x1_ = x0_ + w_
    y1_ = y0_ + h_
    final_keypoint = None 
    for j, keypoint in enumerate(keypoints):
        if is_keypoint_within_bbox(*bbox, keypoint):
            final_keypoint = keypoint
            keypoints = np.delete(keypoints, j, axis=0)
            break
    if final_keypoint is None:
        return None  
    orig_keypoint = final_keypoint.copy()
    # Shift and scale original keypoints
    final_keypoint[0, :] -= x0_
    final_keypoint[1, :] -= y0_
    final_keypoint /= w_
    final_keypoint = np.array([final_keypoint[j, i] for i in range(final_keypoint.shape[1]) for j in range(2)])
    final_keypoint = torch.from_numpy(final_keypoint).view(1, -1)
    
    # Generator forward pass 
    to_generate = preprocess_images(to_generate, 1.0).cuda()
    to_generate = generator(to_generate, final_keypoint)
    to_generate = denormalize_img(to_generate)

    # Post-process generated image
    to_generate = image_to_numpy(to_generate.detach().cpu())[0]
    to_generate = (to_generate * 255).astype("uint8")
    if resize:
        to_generate = cv2.resize(to_generate, (h_, w_))
    return to_generate, orig_keypoint


def anonymize_image(image, keypoints, bounding_boxes, generator, imsize, anonymize_single_bbox):
    image = image.copy()
    keypoints = keypoints.copy()
    replaced_mask = np.ones_like(image).astype("bool")
    replaced_bboxes = []
    replaced_keypoints = []
    
    for bbox in bounding_boxes:
        try:
            x0_, y0_, w_, h_ = expand_bounding_box(*bbox, 0.35, image.shape)
        except AssertionError as e:
            print("Could not process image, bbox error", e)
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
        x0, y0, x1, y1 = bbox
        replaced_mask_cut[y0:y1, x0:x1, :] = 0
    #image = draw_bboxes(image, replaced_bboxes, (0, 0, 255))
    #image = draw_keypoints(image, replaced_keypoints, (0, 0, 255))
    return image


def get_all_bounding_boxes(opts):
    #train = get_bounding_boxes(opts, "train")
    val = get_bounding_boxes(opts, "val")
    #test = get_bounding_boxes(opts, "test")
    res = { **val}#, **test}
    return res 

def get_bounding_boxes(opts, dataset):
    """
        Reads annotated bounding boxes for WIDER FACE dataset

        Returns: (dict) of bounding boxes of shape
            {
                relative_image_path: [x0, y0, x1, y1]
            }
    """
    assert dataset in ["val", "train", "test"]
    relative_im_path = "WIDER_{}/images/".format(dataset)
    if dataset == "test":
        relative_bbox_path = "wider_face_split/wider_face_test_filelist.txt"
    else:
        relative_bbox_path = "wider_face_split/wider_face_{}_bbx_gt.txt".format(dataset)
    total_path = os.path.join(opts.source_dir, relative_bbox_path)
    assert os.path.isfile(total_path), "Did not find annotations in path:" \
                                       + total_path

    with open(total_path, "r") as f:
        lines = list(f.readlines())
    idx = 0 #lines
    bounding_boxes = {

    }
    while idx < len(lines):
        filename = lines[idx].strip()
        idx += 1
        num_bbox = int(lines[idx])
        idx += 1
        filepath = os.path.join(relative_im_path, filename)

        bounding_boxes[filepath] = []
        invalid_image = False
        for i in range(num_bbox):
            # x1, y1, w, h, blur,expression,illumination,invalid,occlusion,pose
            line = [int(x) for x in lines[idx].strip().split(" ")]
            idx += 1
            if line[6] == 1:
                #print("Invalid image:", line, filename)
                invalid_image = True
            #assert line[6] == 0, "Image is invalid"
            x0, y0, w, h = line[:4]
            if w == 0 or h == 0:
                invalid_image = True
            x1 = x0 + w
            y1 = y0 + h
            if  w != 0 and h != 0:
                bounding_boxes[filepath].append([x0, y0, x1, y1])
        #if invalid_image:
            #del bounding_boxes[filepath]
            #bounding_boxes[filepath] = []
    return bounding_boxes


def get_detectron_keypoint(total_filepath, threshold):
    filename = total_filepath.replace("/", "-")
    os.makedirs(os.path.join("scripts", ".detectron_predictions"), exist_ok=True)
    prediction_path = os.path.join("scripts", ".detectron_predictions", filename + "_{}.npy".format(threshold))
    if os.path.isfile(prediction_path):
        return np.load(prediction_path)
    image_keypoints = predict_keypoint(total_filepath, threshold)
    filedir = os.path.dirname(prediction_path)
    os.makedirs(os.path.join("scripts", "detectron_predictions"), exist_ok=True)
    np.save(prediction_path, image_keypoints)
    return image_keypoints


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--source-dir", dest="source_dir",
                    help="Set the source directory of dataset")
    parser.add_option("--target-dir", dest="target_dir",
                    help="Set the directory to save anonymized dataset")
    parser.add_option("--model-name", dest="model_name",
                    help="Set the model name to use for anonymization")

    opts, _ = parser.parse_args()
    # Initialize model
    ckpt_path = os.path.join("checkpoints", opts.model_name)
    ckpt = load_checkpoint(ckpt_path)

    imsize = ckpt["current_imsize"]
    bounding_boxes = get_all_bounding_boxes(opts)
    generator = init_generator(ckpt)

    for filepath in tqdm.tqdm(bounding_boxes.keys()):
        
        total_filepath = os.path.join(opts.source_dir, filepath)
        im = cv2.imread(total_filepath).copy() # BGR
        im = im[:, :, ::-1] # BGR to RGB
        im_bounding_boxes = bounding_boxes[filepath]
        image_keypoints = get_detectron_keypoint(total_filepath, 0)#
        np.save("test.npy", image_keypoints)
        if image_keypoints is None or len(image_keypoints) == 0 or image_keypoints.tolist() is None:
            new_filepath = os.path.join(opts.target_dir, filepath)
            cv2.imwrite(new_filepath, im[:, :, ::-1])
            continue
        image_keypoints = image_keypoints[:, :2, :ckpt["pose_size"]]
        anonymized_image = anonymize_image(im,
                                        image_keypoints,
                                        im_bounding_boxes,
                                        generator,
                                        imsize,
                                        anonymize_single_bbox)
        new_filepath = os.path.join(opts.target_dir, filepath)
        print("Saving to:", new_filepath)
        filedir = os.path.dirname(new_filepath)
        os.makedirs(filedir, exist_ok=True)
        cv2.imwrite(new_filepath, anonymized_image[:, :, ::-1])
