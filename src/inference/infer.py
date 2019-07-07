import os
import cv2
import torch
import numpy as np
from src import config_parser, utils
from src.models.generator import Generator
from src.detection.detection_api import detect_faces_with_keypoints
from src.dataset_tools.utils import expand_bounding_box
from src.data_tools.dataloaders_v2 import cut_bounding_box
from src.data_tools.data_utils import denormalize_img
from .utils import to_torch, image_to_numpy
from src.visualization import utils as vis_utils

def init_generator(config, ckpt):
    g = Generator(
        config.models.pose_size,
        config.models.start_channel_size,
        config.models.image_channels
    )
    g.load_state_dict(ckpt["G"])
    utils.to_cuda(g)
    return g

def get_images_recursive(image_folder):
    endings = [".jpg", ".jpeg", ".png"]
    if os.path.isfile(image_folder):
        return [image_folder]
    images = []
    for root, dirs, files in os.walk(image_folder):
        for fname in files:
            if fname[-4:] in endings or fname[-5:] in endings:
                impath = os.path.join(
                    root, fname
                )
                images.append(impath)
    return images

def shift_bbox(orig_bbox, expanded_bbox, new_imsize):
    x0, y0, x1, y1 = orig_bbox
    x0e, y0e, x1e, y1e = expanded_bbox
    x0, x1 = x0 - x0e, x1 - x1e
    y0, y1 = y0 - y0e, y1 - y1e
    w_ = x1 - x0
    x0, y0, x1, y1 = [int(k/w_*new_imsize) for k in [x0, y0, x1, y1]]
    return [x0, y0, x1, y1]

def keypoint_to_torch(keypoint):
    keypoint = np.array([keypoint[i, j] for j in range(2) for i in range(keypoint.shape[0]) ])
    keypoint = torch.from_numpy(keypoint).view(1, -1)
    return keypoint

def shift_and_scale_keypoint(keypoint, expanded_bbox):
    keypoint = keypoint.copy()
    keypoint[:, 0] -= expanded_bbox[0]
    keypoint[:, 1] -= expanded_bbox[1]
    w = expanded_bbox[2]
    keypoint /= w
    return keypoint
    
def anonymize_face(im, keypoints, bbox, generator, imsize):
    to_generate = cv2.resize(im, (imsize, imsize))
    to_generate = cut_bounding_box(to_generate, bbox)
    to_generate = to_torch(to_generate)
    keypoints = keypoint_to_torch(keypoints)
    to_generate = to_generate * 2 - 1
    with torch.no_grad():
        to_generate = generator(to_generate, keypoints)
        to_generate = denormalize_img(to_generate)
    to_generate = image_to_numpy(to_generate[0])
    to_generate = (to_generate*255).astype(np.uint8)
    to_generate = cv2.resize(to_generate, (im.shape[0], im.shape[1]))
    return to_generate


def anonymize_image(im, keypoints, bounding_boxes, generator, imsize):
    im = im.copy()
    if len(keypoints) == 0:
        return im.copy()
    replaced_mask = np.ones_like(im).astype("bool")
    for keypoint, bbox in zip(keypoints, bounding_boxes):
        try:
            x0e,  y0e, we, he = expand_bounding_box(*bbox, 0.35, im.shape)
            x1e = x0e + we
            y1e = y0e + he
        except AssertionError as e:
            print("Could not process image, bbox error", e)
            continue 
        to_replace = im[y0e:y1e, x0e:x1e]
        new_bbox = shift_bbox(bbox, [x0e, y0e, x1e, y1e], imsize)
        neW_keypoint = shift_and_scale_keypoint(keypoint, [x0e, y0e, x1e, y1e])
        generated_face = anonymize_face(to_replace, keypoint, new_bbox, generator, imsize)
        
        mask_single_face = replaced_mask[y0e:y1e, x0e:x1e]
        to_replace[mask_single_face] = generated_face[mask_single_face]
        im[y0e:y1e, x0e:x1e] = to_replace
        x0, y0, x1, y1 = bbox
        mask_single_face[y0:y1, x0:x1] = 0
    return im


if __name__ == "__main__":
    
    config = config_parser.initialize_and_validate_config([
        {"name": "source_path", "default": "test_examples/source"},
        {"name": "target_path", "default": ""}
    ])
    save_path = config.target_path
    if save_path == "":
        default_path = os.path.join(
            os.path.dirname(config.config_path),
            "anonymized_images"
        )
        print("Setting target path to default:", default_path)
        save_path = default_path
    ckpt = utils.load_checkpoint(config.checkpoint_dir)
    generator = init_generator(config, ckpt)

    imsize = ckpt["current_imsize"]
    image_paths = get_images_recursive(config.source_path)
    for filepath in image_paths:
        im = cv2.imread(filepath)
        face_boxes, keypoints = detect_faces_with_keypoints(im)
        anonymized_image = anonymize_image(im[:, :,::-1], keypoints, face_boxes, generator, imsize)
        annotated_im = vis_utils.draw_faces_with_keypoints(im, face_boxes, keypoints)
        to_save = np.concatenate((annotated_im, anonymized_image[:, :, ::-1]), axis=1)

        relative_path = filepath[len(config.source_path)+1:]
        
        im_savepath = os.path.join(save_path, relative_path)
        print("Saving to:", im_savepath)
        os.makedirs(os.path.dirname(im_savepath), exist_ok=True)
        cv2.imwrite(im_savepath, to_save)
