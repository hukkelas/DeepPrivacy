import os
import cv2
import torch
import numpy as np
from ..data_tools import data_utils
from src import config_parser, utils
from src.models.generator import Generator
from src.detection.detection_api import detect_faces_with_keypoints
from src.dataset_tools import utils as dataset_utils
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
    g.eval()
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


def to_numpy(el):
    if isinstance(el, torch.Tensor):
        return el.numpy()
    if isinstance(el, list):
        return np.array(el)
    return el


def shift_bbox(orig_bbox, expanded_bbox, new_imsize):
    orig_bbox = to_numpy(orig_bbox).astype(float)
    expanded_bbox = to_numpy(expanded_bbox).astype(float)

    x0, y0, x1, y1 = orig_bbox
    x0e, y0e, x1e, y1e = expanded_bbox
    x0, x1 = x0 - x0e, x1 - x0e
    y0, y1 = y0 - y0e, y1 - y0e
    w_ = x1e - x0e
    x0, y0, x1, y1 = [int(k*new_imsize/w_) for k in [x0, y0, x1, y1]]
    bbox = np.array([x0, y0, x1, y1]).astype(int)
    return [x0, y0, x1, y1]


def keypoint_to_torch(keypoint):
    keypoint = np.array([keypoint[i, j] for i in range(keypoint.shape[0]) for j in range(2)])
    keypoint = torch.from_numpy(keypoint).view(1, -1)
    return keypoint


def keypoint_to_numpy(keypoint):
    return keypoint.view(-1, 2).numpy()


def shift_and_scale_keypoint(keypoint, expanded_bbox):
    keypoint = keypoint.copy().astype(float)
    keypoint[:, 0] -= expanded_bbox[0]
    keypoint[:, 1] -= expanded_bbox[1]
    w = expanded_bbox[2] - expanded_bbox[0]
    keypoint /= w
    return keypoint


def save_debug_image(original_image, input_image, generated, keypoints, bbox, expanded_bbox):
    x0e, y0e, x1e, y1e = expanded_bbox
    original_image = original_image[y0e:y1e, x0e:x1e]
    original_image = cv2.resize(original_image, (input_image.shape[0], input_image.shape[0]))

    bbox = np.array(bbox)
    imname = os.path.basename(filepath).split(".")[0]

    debug_path = os.path.join(save_path, "debug")

    os.makedirs(debug_path, exist_ok=True)
    debug_path = os.path.join(debug_path, f"{imname}_{face_idx}.jpg")
    x = keypoints[0, range(0, len(keypoints[0]), 2)]
    y = keypoints[0, range(1, len(keypoints[0]), 2)]
    keypoints = (torch.stack((x, y), dim=1) * original_image.shape[0])[None, :]
    original_image = vis_utils.draw_faces_with_keypoints(original_image,
                                                         bbox[None, :],
                                                         keypoints,
                                                         draw_bboxes=False)
    image = np.concatenate((original_image, input_image, generated), axis=1)
    cv2.imwrite(debug_path, image[:, :, ::-1])


def anonymize_face(im, keypoints, generator):
    with torch.no_grad():
        generated = generator(im, keypoints)
    return generated


def pre_process(im, keypoint, bbox, imsize, cuda=True):
    bbox = to_numpy(bbox)
    try:
        expanded_bbox = dataset_utils.expand_bbox_simple(bbox, 0.4)
    except AssertionError as e:
        print("Could not process image, bbox error", e)
        return None
    to_replace = dataset_utils.pad_image(im, expanded_bbox)
    new_bbox = shift_bbox(bbox, expanded_bbox, imsize)
    new_keypoint = shift_and_scale_keypoint(keypoint, expanded_bbox)
    to_replace = cv2.resize(to_replace, (imsize, imsize))
    to_replace = cut_bounding_box(to_replace.copy(), torch.tensor(new_bbox), 1.0)
    to_replace = torch_utils.image_to_torch(to_replace, cuda=cuda)
    keypoint = keypoint_to_torch(new_keypoint)
    torch_input = to_replace * 2 - 1
    return torch_input, keypoint, expanded_bbox, new_bbox


def stitch_face(im, expanded_bbox, generated_face, bbox_to_extract, image_mask, original_bbox):
    # Ugly but works....
    x0e, y0e, x1e, y1e = expanded_bbox
    x0o, y0o, x1o, y1o = bbox_to_extract

    mask_single_face = image_mask[y0e:y1e, x0e:x1e]
    to_replace = im[y0e:y1e, x0e:x1e]
    generated_face = generated_face[y0o:y1o, x0o:x1o]
    to_replace[mask_single_face] = generated_face[mask_single_face]
    im[y0e:y1e, x0e:x1e] = to_replace
    x0, y0, x1, y1 = original_bbox
    image_mask[y0:y1, x0:x1, :] = 0
    return im


def replace_face(im, generated_face, image_mask, original_bbox):
    original_bbox = to_numpy(original_bbox)
    expanded_bbox = dataset_utils.expand_bbox_simple(original_bbox, 0.4)
    assert expanded_bbox[2] - expanded_bbox[0] == generated_face.shape[1], f'Was: {expanded_bbox}, Generated Face: {generated_face.shape}'
    assert expanded_bbox[3] - expanded_bbox[1] == generated_face.shape[0], f'Was: {expanded_bbox}, Generated Face: {generated_face.shape}'

    bbox_to_extract = np.array([0, 0, generated_face.shape[1], generated_face.shape[0]])
    for i in range(2):
        if expanded_bbox[i] < 0:
            bbox_to_extract[i] -= expanded_bbox[i]
            expanded_bbox[i] = 0
    if expanded_bbox[2] > im.shape[1]:
        diff = expanded_bbox[2] - im.shape[1]
        bbox_to_extract[2] -= diff
        expanded_bbox[2] = im.shape[1]
    if expanded_bbox[3] > im.shape[0]:
        diff = expanded_bbox[3] - im.shape[0]
        bbox_to_extract[3] -= diff
        expanded_bbox[3] = im.shape[0]

    im = stitch_face(im, expanded_bbox, generated_face, bbox_to_extract, image_mask, original_bbox)
    return im


def post_process(im, generated_face, expanded_bbox, original_bbox, image_mask):
    generated_face = denormalize_img(generated_face)
    generated_face = torch_utils.image_to_numpy(generated_face[0], to_uint8=True)
    orig_imsize = expanded_bbox[2] - expanded_bbox[0]
    generated_face = cv2.resize(generated_face, (orig_imsize, orig_imsize))
    im = replace_face(im, generated_face, image_mask, original_bbox)
    return im


def anonymize_image(im, keypoints, bounding_boxes, generator, imsize, verbose=False):
    im = im.copy()
    if len(keypoints) == 0:
        return im.copy()
    replaced_mask = np.ones_like(im).astype("bool")
    global face_idx
    face_idx = 0
    for keypoint, original_bbox in zip(keypoints, bounding_boxes):
        res = pre_process(im, keypoint, original_bbox, imsize)
        if res is None:
            continue
        to_replace, keypoint, expanded_bbox, shifted_bbox = res
        generated_face = anonymize_face(to_replace, keypoint, generator)
        im = post_process(im, generated_face, expanded_bbox, original_bbox, replaced_mask)
        face_idx += 1
        if verbose:
            save_debug_image(
                im,
                torch_utils.image_to_numpy(data_utils.denormalize_img(to_replace)[0], to_uint8=True),
                torch_utils.image_to_numpy(data_utils.denormalize_img(generated_face)[0], to_uint8=True),
                keypoint,
                shifted_bbox,
                expanded_bbox
            )
    return im


def read_args(additional_args=[]):
    config = config_parser.initialize_and_validate_config([
        {"name": "source_path", "default": "test_examples/source"},
        {"name": "target_path", "default": ""}
    ] + additional_args)
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
    source_path = config.source_path
    image_paths = get_images_recursive(source_path)
    if additional_args:
        return generator, imsize, source_path, image_paths, save_path, config
    return generator, imsize, source_path, image_paths, save_path
