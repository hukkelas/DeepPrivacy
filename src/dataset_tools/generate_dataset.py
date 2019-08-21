import os
import tqdm
import cv2
import multiprocessing
import torch
import math
import argparse
import numpy as np
from PIL import Image
from src.dataset_tools.utils import (expand_bounding_box, read_json,
                                     is_keypoint_within_bbox,
                                     expand_bbox_simple, pad_image)

TARGET_DIR = "data/test"
IMAGE_TARGET_DIR = os.path.join(TARGET_DIR, "original")
os.makedirs(IMAGE_TARGET_DIR, exist_ok=True)
BBOX_TARGET_DIR = os.path.join(TARGET_DIR, "bounding_box")
os.makedirs(BBOX_TARGET_DIR, exist_ok=True)
LANDMARK_TARGET_DIR = os.path.join(TARGET_DIR, "landmarks")
os.makedirs(LANDMARK_TARGET_DIR, exist_ok=True)

np.random.seed(0)
IMAGE_SOURCE_DIR = "/work/haakohu/yfcc100m/images2"
LANDMARKS_PATH = "/lhome/haakohu/flickr_download/annotations_keypoints.json"
BBOX_PATH = "/lhome/haakohu/flickr_download/annotations.json"
BBOX_JSON = read_json(BBOX_PATH)
LANDMARKS_JSON = read_json(LANDMARKS_PATH)
BBOX_EXPANSION_FACTOR = 0.35
MIN_BBOX_SIZE = 128
parser = argparse.ArgumentParser()
parser.add_argument("--max_imsize", default=128, type=int)
parser.add_argument("--min_imsize", default=4, type=int)
parser.add_argument("--simple_expand", default=True, action="store_false",
                    help="Expands the face bounding box from the center. Can include black borders.")
args = parser.parse_args()
if args.simple_expand:
    BBOX_EXPANSION_FACTOR = .4
num_sizes = int(math.log(args.max_imsize/args.min_imsize, 2))
TARGET_IMSIZES = [args.min_imsize * (2**k) for k in range(1, num_sizes+1)]


def get_imnames():
    imnames1 = set(LANDMARKS_JSON.keys())
    imnames2 = set(BBOX_JSON.keys())
    image_names = list(imnames2.intersection(imnames1))
    image_names.sort()
    return image_names


def match_bbox_keypoint(bounding_boxes, keypoints):
    """
        bounding_boxes shape: [N, 5]
        keypoints: [N persons, (X, Y, Score, ?), K Keypoints]
    """
    if len(bounding_boxes) == 0 or len(keypoints) == 0:
        return None, None
    assert bounding_boxes.shape[1] == 5, "Shape was : {}".format(bounding_boxes.shape)
    assert keypoints.shape[1:] == (4, 7), "Keypoint shape was: {}".format(keypoints.shape)
    # Sort after score
    sorted_idx = np.argsort(bounding_boxes[:, 4])[::-1]
    bounding_boxes = bounding_boxes[sorted_idx]

    matches = []
    bounding_boxes = bounding_boxes[:, :4]
    keypoints = keypoints[:, :2]
    for bbox_idx, bbox in enumerate(bounding_boxes):
        keypoint = None
        for kp_idx, keypoint in enumerate(keypoints):
            if kp_idx in [x[1] for x in matches]:
                continue
            if is_keypoint_within_bbox(*bbox, keypoint):
                matches.append((bbox_idx, kp_idx))
                break
    keypoint_idx = [x[1] for x in matches]
    bbox_idx = [x[0] for x in matches]
    return bounding_boxes[bbox_idx], keypoints[keypoint_idx]


def process_face(bbox, landmark, imshape):
    assert bbox.shape == (4,), "Was shape: {}".format(bbox.shape)
    assert landmark.shape == (2, 7), "Was shape: {}".format(landmark.shape)
    if args.simple_expand:
        x0, y0, x1, y1 = expand_bbox_simple(bbox, BBOX_EXPANSION_FACTOR)
    else:
        try:
            x0, y0, w, h = expand_bounding_box(*bbox, BBOX_EXPANSION_FACTOR, imshape)
            x1 = x0 + w
            y1 = y0 + h
            x0, y0, x1, y1 = [int(_) for _ in [x0, y0, x1, y1]]
        except AssertionError:
            return None
    width = x1 - x0
    if width < MIN_BBOX_SIZE:
        return None
    bbox[[0, 2]] -= x0
    bbox[[1, 3]] -= y0
    assert width == y1 - y0, f"width: {width}, height: {y1-y0}"
    bbox = bbox.astype("int")
    landmark[0] -= x0
    landmark[1] -= y0
    landmark = np.array([landmark[j, i] for i in range(landmark.shape[1]) for j in range(2)])
    return {
        "expanded_bbox": np.array([x0, y0, x1, y1]),
        "face_bbox": bbox,
        "landmark": landmark.flatten()
    }


def process_image(imname):
    impath = os.path.join(IMAGE_SOURCE_DIR, imname)
    bounding_boxes = np.array(BBOX_JSON[imname])
    landmarks = np.array(LANDMARKS_JSON[imname]["cls_keyps"])
    bounding_boxes, landmarks = match_bbox_keypoint(bounding_boxes, landmarks)
    if bounding_boxes is None:
        return [], impath
    assert bounding_boxes.shape[0] == landmarks.shape[0]

    im = Image.open(impath)

    imshape = im.size
    imshape = (imshape[1], imshape[0], *imshape[2:])
    resulting_annotation = []
    for bbox, landmark in zip(bounding_boxes, landmarks):
        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        bbox[2] = min(imshape[1], bbox[2])
        bbox[3] = min(imshape[0], bbox[3])
        face_res = process_face(bbox.copy(), landmark, imshape)
        if face_res is not None:
            resulting_annotation.append(face_res)
    return resulting_annotation, impath


def pool(img):
    img = img.astype(np.float32)
    img = (img[0::2, 0::2] + img[0::2, 1::2] + img[1::2, 0::2] + img[1::2, 1::2]) * 0.25
    img = img.astype(np.uint8)
    return img


def extract_and_save_image_batch(impaths, image_annotations, batch_idx):
    images = []
    for impath in impaths:
        images.append(np.array(Image.open(impath).convert("RGB")))
    extracted_faces = []
    for image, annotations in zip(images, image_annotations):
        for annotation in annotations:
            x0, y0, x1, y1 = annotation["expanded_bbox"]
            if args.simple_expand:
                cut_im = pad_image(image, [x0, y0, x1, y1])
            else:
                cut_im = image[y0:y1, x0:x1]
            extracted_faces.append(cut_im)
    max_imsize = TARGET_IMSIZES[-1]
    extracted_faces = [
        cv2.resize(im, (max_imsize, max_imsize), interpolation=cv2.INTER_AREA)
        for im in extracted_faces
    ]
    to_save = [np.zeros((len(extracted_faces), imsize, imsize, 3), dtype=np.uint8)
               for imsize in TARGET_IMSIZES]
    for im_idx, im in enumerate(extracted_faces):
        for imsize_idx in range(len(TARGET_IMSIZES)-1, -1, -1):
            assert im.shape == (TARGET_IMSIZES[imsize_idx], TARGET_IMSIZES[imsize_idx], 3), f'Imsize was: {im.shape}'
            to_save[imsize_idx][im_idx] = im
            im = pool(im)
    for faces, imsize in zip(to_save, TARGET_IMSIZES):
        assert faces.shape[1:] == (imsize, imsize, 3), f'Shape was: {faces.shape}'
        target_dir = os.path.join(IMAGE_TARGET_DIR, str(imsize))
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, f"{batch_idx}.npy")
        np.save(target_path, faces)


def save_annotation(bounding_boxes, landmarks, sizes):
    normalized_bbox = bounding_boxes
    normalized_landmark = landmarks

    for imsize in TARGET_IMSIZES:
        bbox_to_save = normalized_bbox / sizes * imsize
        bbox_to_save = torch.from_numpy(bbox_to_save).long()

        assert bbox_to_save.shape == bounding_boxes.shape

        target_path = os.path.join(BBOX_TARGET_DIR, "{}.torch".format(imsize))
        torch.save(bbox_to_save, target_path)

        landmark_to_save = normalized_landmark / sizes * imsize
        landmark_to_save = torch.from_numpy(landmark_to_save)

        target_path = os.path.join(LANDMARK_TARGET_DIR, "{}.torch".format(imsize))
        torch.save(landmark_to_save, target_path)


def extract_annotations_and_save(image_annotations):
    bounding_boxes = []
    landmarks = []
    sizes = []
    for annotations in tqdm.tqdm(image_annotations, desc="Saving annotations"):
        for annotation in annotations:
            bounding_boxes.append(annotation["face_bbox"])
            landmarks.append(annotation["landmark"])
            x0, y0, x1, y1 = annotation["expanded_bbox"]
            assert int(y1 - y0) == int(x1 - x0), "Expected image to have equal sizes. Was: {}, {}".format(x1- x0, y1 - y0)
            sizes.append(y1 - y0)
    bounding_boxes = np.stack(bounding_boxes, axis=0)
    landmarks = np.stack(landmarks, axis=0)
    sizes = np.array(sizes).reshape(-1, 1)
    save_annotation(bounding_boxes, landmarks, sizes)


def main():
    image_names = get_imnames()
    impaths = []
    image_annotations = []
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        for imname in image_names:
            job = pool.apply_async(process_image, (imname, ))
            jobs.append(job)
        for job in tqdm.tqdm(jobs, desc="Pre-processing annotations."):
            annotation, impath = job.get()
            impaths.append(impath)
            image_annotations.append(annotation)
    extract_annotations_and_save(image_annotations)
    total_images = [len(x) for x in image_annotations]
    print("Total number of images:", sum(total_images))
    num_jobs = 20000
    batch_size = math.ceil(len(impaths) / num_jobs)
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        for i in range(num_jobs):
            impath = impaths[i*batch_size:(i+1)*batch_size]
            annotations = image_annotations[i*batch_size:(i+1)*batch_size]
            job = pool.apply_async(extract_and_save_image_batch, (impath, annotations, i))
            jobs.append(job)
        for job in tqdm.tqdm(jobs):
            job.get()


if __name__ == "__main__":
    main()
