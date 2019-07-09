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
import src.torch_utils as torch_utils
from src.visualization import utils as vis_utils

def init_generator(config, ckpt):
    g = Generator(
        config.models.pose_size,
        config.models.start_channel_size,
        config.models.image_channels
    )
    g.load_state_dict(ckpt["G"])
    torch_utils.to_cuda(g)
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
    x0, x1 = x0 - x0e, x1 - x0e
    y0, y1 = y0 - y0e, y1 - y0e
    w_ = x1e - x0e
    x0, y0, x1, y1 = [int(k/w_*new_imsize) for k in [x0, y0, x1, y1]]
    return [x0, y0, x1, y1]

def keypoint_to_torch(keypoint):
    keypoint = np.array([keypoint[i, j] for i in range(keypoint.shape[0]) for j in range(2) ])
    keypoint = torch.from_numpy(keypoint).view(1, -1)
    return keypoint

def shift_and_scale_keypoint(keypoint, expanded_bbox):
    keypoint = keypoint.copy()
    keypoint[:, 0] -= expanded_bbox[0]
    keypoint[:, 1] -= expanded_bbox[1]
    w = expanded_bbox[2] - expanded_bbox[0]
    keypoint /= w
    return keypoint

def save_debug_image(original_image, input_image, generated, keypoints, bbox):
    bbox = np.array(bbox)
    imname = os.path.basename(filepath).split(".")[0]

    debug_path = os.path.join(save_path, "debug")

    os.makedirs(debug_path, exist_ok=True)
    debug_path = os.path.join(debug_path, f"{imname}_{face_idx}.jpg")
    x = keypoints[0, range(0, len(keypoints[0]), 2)]
    y = keypoints[0, range(1, len(keypoints[0]), 2)]
    keypoints = (torch.stack((x,y), dim=1) * original_image.shape[0])[None, :]
    original_image = vis_utils.draw_faces_with_keypoints(original_image,
                                                         bbox[None, :],
                                                         keypoints,
                                                         draw_bboxes=False)
    image = np.concatenate((original_image, input_image, generated), axis=1)
    cv2.imwrite(debug_path, image[:, :, ::-1])

def anonymize_face(im, keypoints, bbox, generator, imsize, verbose=False):
    resized_im = cv2.resize(im, (imsize, imsize))
    to_generate = cut_bounding_box(resized_im.copy(), bbox)
    
    to_generate = torch_utils.image_to_torch(to_generate)
    keypoints = keypoint_to_torch(keypoints)
    torch_input = to_generate * 2 - 1
    with torch.no_grad():
        generated = generator(torch_input, keypoints)
        generated = denormalize_img(generated)
    generated = torch_utils.image_to_numpy(generated[0])
    
    generated = (generated*255).astype(np.uint8)

    if verbose:
        save_debug_image(resized_im, torch_utils.image_to_numpy((torch_input[0]+1)/2, to_uint8=True),
                         generated, keypoints, bbox)
    to_generate = cv2.resize(generated, (im.shape[0], im.shape[1]))
    return to_generate


def anonymize_image(im, keypoints, bounding_boxes, generator, imsize, verbose=False):
    im = im.copy()
    if len(keypoints) == 0:
        return im.copy()
    replaced_mask = np.ones_like(im).astype("bool")
    global face_idx
    face_idx = 0
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
        new_keypoint = shift_and_scale_keypoint(keypoint, [x0e, y0e, x1e, y1e])
        generated_face = anonymize_face(to_replace, new_keypoint, new_bbox, generator, imsize, verbose)
        
        mask_single_face = replaced_mask[y0e:y1e, x0e:x1e]
        to_replace[mask_single_face] = generated_face[mask_single_face]
        im[y0e:y1e, x0e:x1e] = to_replace
        x0, y0, x1, y1 = bbox
        mask_single_face[y0:y1, x0:x1] = 0
        face_idx += 1
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
        face_boxes, keypoints = detect_faces_with_keypoints(im, keypoint_threshold=0.5)
        anonymized_image = anonymize_image(im[:, :,::-1], keypoints, face_boxes, generator, imsize, verbose=True)
        annotated_im = vis_utils.draw_faces_with_keypoints(im, face_boxes, keypoints)
        to_save = np.concatenate((annotated_im, anonymized_image[:, :, ::-1]), axis=1)

        relative_path = filepath[len(config.source_path)+1:]
        
        im_savepath = os.path.join(save_path, relative_path)
        print("Saving to:", im_savepath)
        os.makedirs(os.path.dirname(im_savepath), exist_ok=True)
        cv2.imwrite(im_savepath, to_save)
