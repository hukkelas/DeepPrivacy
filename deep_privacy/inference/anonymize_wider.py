import os
import numpy as np
import shutil
from .infer import read_args
from deep_privacy.inference.deep_privacy_anonymizer import DeepPrivacyAnonymizer
from deep_privacy.inference.blur import PixelationAnonymizer, BlurAnonymizer, BlackOutAnonymizer


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
    idx = 0  # lines
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
            if w != 0 and h != 0:
                bounding_boxes[filepath].append([x0, y0, x1, y1])
    im_bboxes_dict = bounding_boxes
    relative_image_paths = list(im_bboxes_dict.keys())
    im_bboxes = [np.array(im_bboxes_dict[k]) for k in relative_image_paths]
    return relative_image_paths, im_bboxes


def init_anonymizer(anonymization_type, keypoint_threshold, face_threshold, generator):
    if anonymization_type == "deep_privacy":
        return DeepPrivacyAnonymizer(
            generator,
            batch_size=128,
            use_static_z=True,
            keypoint_threshold=keypoint_threshold,
            face_threshold=face_threshold
        )
    elif anonymization_type == "pixelation8":
        return PixelationAnonymizer(
            pixelation_size=8,
            face_threshold=face_threshold,
            keypoint_threshold=keypoint_threshold,
        )
    elif anonymization_type == "pixelation16":
        return PixelationAnonymizer(
            pixelation_size=16,
            face_threshold=face_threshold,
            keypoint_threshold=keypoint_threshold,
        )
    elif anonymization_type == "heavy_blur":
        return BlurAnonymizer(
            "heavy_blur",
            face_threshold=face_threshold,
            keypoint_threshold=keypoint_threshold)
    elif anonymization_type == "gaussian_blur":
        return BlurAnonymizer(
            "gaussian_blur",
            face_threshold=face_threshold,
            keypoint_threshold=keypoint_threshold)
    elif anonymization_type == "black_out":
        return BlackOutAnonymizer(
            "heavy_blur",
            face_threshold=face_threshold,
            keypoint_threshold=keypoint_threshold)
    else:
        raise AttributeError("Did not find anonymization type:",
                             anonymization_type)


if __name__ == "__main__":
    generator, imsize, source_path, image_paths, save_path, config = read_args([
        {"name": "anonymization_type", "default": "deep_privacy"}
    ])

    relative_image_paths, im_bboxes = get_bounding_boxes(source_path, "val")

    image_paths = [os.path.join(source_path, k) for k in relative_image_paths]
    face_threshold = 0.0
    keypoint_threshold = 0.01

    anonymizer = init_anonymizer(config.anonymization_type,
                                 keypoint_threshold,
                                 face_threshold,
                                 generator)
    save_paths = [os.path.join(save_path, relative_path)
                  for relative_path in relative_image_paths]
    anonymizer.anonymize_image_paths(image_paths, save_paths,
                                     im_bboxes=im_bboxes)
    # Copy wider_face_val
    source_path = os.path.join(source_path, "wider_face_split")
    target_path = os.path.join(save_path, "wider_face_split")
    if not os.path.isdir(target_path):
        shutil.copytree(source_path, target_path)
