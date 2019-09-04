import os
import cv2
import torch
import numpy as np
from deep_privacy import config_parser, utils
from deep_privacy.models.generator import Generator
from deep_privacy.dataset_tools import utils as dataset_utils
from deep_privacy.data_tools.dataloaders import cut_bounding_box
from deep_privacy.data_tools.data_utils import denormalize_img
import deep_privacy.torch_utils as torch_utils


def init_generator(config, ckpt):
    g = Generator(
        config.models.pose_size,
        config.models.start_channel_size,
        config.models.image_channels
    )
    g.load_state_dict(ckpt["running_average_generator"])
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
    if isinstance(el, list) or isinstance(el, tuple):
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


SIMPLE_EXPAND = False


def pre_process(im, keypoint, bbox, imsize, cuda=True):
    bbox = to_numpy(bbox)
    expanded_bbox = dataset_utils.expand_bbox(bbox, im.shape, SIMPLE_EXPAND,
                                              default_to_simple=True,
                                              expansion_factor1=0.35)
    to_replace = dataset_utils.cut_face(im, expanded_bbox, SIMPLE_EXPAND)
    new_bbox = shift_bbox(bbox, expanded_bbox, imsize)
    new_keypoint = shift_and_scale_keypoint(keypoint, expanded_bbox)
    to_replace = cv2.resize(to_replace, (imsize, imsize))
    to_replace = cut_bounding_box(to_replace.copy(), torch.tensor(new_bbox),
                                  1.0)
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


def replace_face(im, generated_face, image_mask, original_bbox, expanded_bbox):
    original_bbox = to_numpy(original_bbox)
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
    im = replace_face(im, generated_face, image_mask, original_bbox, expanded_bbox)
    return im


def get_default_target_path(source_path, target_path, config_path):
    if target_path != "":
        return target_path
    if source_path.endswith(".mp4"):
        basename = source_path.split(".")
        assert len(basename) == 2
        return f"{basename[0]}_anonymized.mp4"
    default_path = os.path.join(
            os.path.dirname(config_path),
            "anonymized_images"
        )
    print("Setting target path to default:", default_path)
    return default_path


def read_args(additional_args=[]):
    config = config_parser.initialize_and_validate_config([
        {"name": "source_path", "default": "test_examples/source"},
        {"name": "target_path", "default": ""}
    ] + additional_args)
    target_path = config.target_path
    target_path = get_default_target_path(config.source_path,
                                          config.target_path,
                                          config.config_path)
    ckpt = utils.load_checkpoint(config.checkpoint_dir)
    generator = init_generator(config, ckpt)

    imsize = ckpt["current_imsize"]
    source_path = config.source_path
    image_paths = get_images_recursive(source_path)
    if additional_args:
        return generator, imsize, source_path, image_paths, target_path, config
    return generator, imsize, source_path, image_paths, target_path
