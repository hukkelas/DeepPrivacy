import os
import cv2
import numpy as np
import torch
import shutil
import tqdm
from .infer import read_args
from .batch_infer import anonymize_images
from ..detection import detection_api
from deep_privacy.visualization import utils as vis_utils

def get_bounding_boxes(source_dir, dataset):
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
    total_path = os.path.join(source_dir, relative_bbox_path)
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
    return bounding_boxes

def anonymize_images_pixelation(images, im_keypoints, im_bboxes, imsize, generator, batch_size, pixelation_size):
    anonymized_images = []
    for im_idx, im in enumerate(tqdm.tqdm(images, desc="Anonymizing images")):
        anonymized_image = im.copy()
        bboxes = im_bboxes[im_idx]
        for bbox in bboxes:
            x0, y0, x1, y1 = bbox
            face = im[y0:y1, x0:x1]
            face = cv2.resize(face, (pixelation_size, pixelation_size))
            face = cv2.resize(face, (x1 - x0, y1 - y0))
            anonymized_image[y0:y1, x0:x1] = face
        anonymized_images.append(anonymized_image)
    return anonymized_images

def anonymize_images_wider(anonymization_type, *args):
    if anonymization_type == "deep_privacy":
        print("Anonymization type: DeepPrivacy")
        return anonymize_images(*args)
    elif anonymization_type == "pixelation16":
        print("Anonymization type: Pixelation16")
        return anonymize_images_pixelation(*args, 16)
    elif anonymization_type == "pixelation8":
        print("Anonymization type: Pixelation8")
        return anonymize_images_pixelation(*args, 8)
    raise AttributeError


if __name__ == "__main__":
  generator, imsize, source_path, image_paths, save_path, config = read_args([
      {"name": "anonymization_type", "default": "deep_privacy"}
  ])

  im_bboxes_dict = get_bounding_boxes(source_path, "val")
  relative_image_paths = list(im_bboxes_dict.keys())
  im_bboxes = [np.array(im_bboxes_dict[k]) for k in relative_image_paths]
  image_paths = [os.path.join(source_path, k) for k in relative_image_paths]
  images = [cv2.imread(p)[:, :, ::-1] for p in image_paths]
  im_bboxes, im_keypoints = detection_api.batch_detect_faces_with_keypoints(images, im_bboxes=im_bboxes)
  anonymized_images = anonymize_images_wider(config.anonymization_type, 
                                       [i.copy() for i in images],
                                       im_keypoints,
                                       im_bboxes,
                                       imsize,
                                       generator, 128)
  for im_idx in range(len(image_paths)):
    im = images[im_idx]
    anonymized_image = anonymized_images[im_idx]
    filepath = image_paths[im_idx]
    face_boxes, keypoints = im_bboxes[im_idx], im_keypoints[im_idx]
    annotated_im = vis_utils.draw_faces_with_keypoints(im[:, :, ::-1], face_boxes, keypoints)
    to_save = np.concatenate((annotated_im, anonymized_image[:, :, ::-1]), axis=1)

    splitted_filepath = filepath.split(os.sep)
    splitted_source_path = os.path.dirname(source_path).split(os.sep)
    relative_path = splitted_filepath[len(splitted_source_path):]
    relative_path = os.path.join(*relative_path)
    print(relative_path)
    im_savepath = os.path.join(save_path, relative_path)
    print("Saving to:", im_savepath)
    os.makedirs(os.path.dirname(im_savepath), exist_ok=True)
    cv2.imwrite(im_savepath+ "_detection_left_anonymized_right.jpg", to_save)
    cv2.imwrite(im_savepath, anonymized_image[:, :, ::-1])
  # Copy wider_face_val
  source_path = os.path.join(source_path, "wider_face_split")
  target_path = os.path.join(save_path, "wider_face_split")
  if not os.path.isdir(target_path):
    shutil.copytree(source_path, target_path)
