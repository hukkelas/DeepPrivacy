import os
import tqdm
import cv2
import multiprocessing
import torch
import math
import argparse
import numpy as np
import utils
import shutil
from PIL import Image


TARGET_DIR = "data/fdf_new"
shutil.rmtree(TARGET_DIR)
IMAGE_TARGET_DIR = os.path.join(TARGET_DIR, "images")
os.makedirs(IMAGE_TARGET_DIR)
BBOX_TARGET_DIR = os.path.join(TARGET_DIR, "bounding_box")
os.makedirs(BBOX_TARGET_DIR)
LANDMARK_TARGET_DIR = os.path.join(TARGET_DIR, "landmarks")
os.makedirs(LANDMARK_TARGET_DIR)

np.random.seed(0)
IMAGE_SOURCE_DIR = "/work/haakohu/yfcc100m/images2"
#LANDMARKS_PATH = "/lhome/haakohu/flickr_download/annotations_keypoints.json"
LANDMARKS_PATH = "test_keypoints.json"

#BBOX_PATH = "/lhome/haakohu/flickr_download/annotations.json"
BBOX_PATH = "test_bbox.json"
BBOX_JSON = utils.read_json(BBOX_PATH)
LANDMARKS_JSON = utils.read_json(LANDMARKS_PATH)
fdf_metainfo = utils.read_json("fdf_metainfo.json")

MIN_BBOX_SIZE = 128
parser = argparse.ArgumentParser()
parser.add_argument("--max_imsize", default=128, type=int)
parser.add_argument("--min_imsize", default=4, type=int)
parser.add_argument("--simple_expand", default=False, action="store_true",
                    help="Expands the face bounding box from the center. Can include black borders.")
args = parser.parse_args()


num_sizes = int(math.log(args.max_imsize/args.min_imsize, 2))
TARGET_IMSIZES = [args.min_imsize * (2**k) for k in range(1, num_sizes+1)]

for imsize in TARGET_IMSIZES:
    folder = os.path.join(IMAGE_TARGET_DIR, str(imsize))
    os.makedirs(folder)


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
            if utils.is_keypoint_within_bbox(*bbox, keypoint):
                matches.append((bbox_idx, kp_idx))
                break
    keypoint_idx = [x[1] for x in matches]
    bbox_idx = [x[0] for x in matches]
    return bounding_boxes[bbox_idx], keypoints[keypoint_idx]


def process_face(bbox, landmark, imshape, imname):
    assert bbox.shape == (4,), "Was shape: {}".format(bbox.shape)
    assert landmark.shape == (2, 7), "Was shape: {}".format(landmark.shape)
    orig_bbox = bbox.copy()
    orig_landmark = landmark.copy()
    expanded_bbox = utils.expand_bbox(bbox, imshape, args.simple_expand)
    if expanded_bbox is None:
        return None

    width = expanded_bbox[2] - expanded_bbox[0]
    height = expanded_bbox[3] - expanded_bbox[1]
    if width < MIN_BBOX_SIZE:
        return None
    bbox[[0, 2]] -= expanded_bbox[0]
    bbox[[1, 3]] -= expanded_bbox[1]
    assert width == height, f"width: {width}, height: {y1-y0}"
    bbox = bbox.astype("int")
    landmark[0] -= expanded_bbox[0]
    landmark[1] -= expanded_bbox[1]
    landmark = np.array([landmark[j, i] for i in range(landmark.shape[1]) for j in range(2)])
    return {
        "expanded_bbox": expanded_bbox,
        "face_bbox": bbox,
        "landmark": landmark.flatten(),
        "orig_bbox": orig_bbox,
        "orig_landmark": orig_landmark,
        "line_idx": imname.split(".")[0]
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
        face_res = process_face(bbox.copy(), landmark, imshape, imname)
        if face_res is not None:
            resulting_annotation.append(face_res)
    return resulting_annotation, impath


def pool(img):
    img = img.astype(np.float32)
    img = (img[0::2, 0::2] + img[0::2, 1::2] + img[1::2, 0::2] + img[1::2, 1::2]) * 0.25
    img = img.astype(np.uint8)
    return img


def save_face(original_im, face_annotation, im_idx):
    im = utils.cut_face(original_im, face_annotation["expanded_bbox"],
                        args.simple_expand)
    max_imsize = TARGET_IMSIZES[-1]
    im = cv2.resize(im, (max_imsize, max_imsize), interpolation=cv2.INTER_AREA)

    for imsize_idx in range(len(TARGET_IMSIZES)-1, -1, -1):
        imsize = TARGET_IMSIZES[imsize_idx]
        assert im.shape == (imsize, imsize, 3)
        assert im.dtype == np.uint8
        impath = os.path.join(IMAGE_TARGET_DIR, str(imsize), f'{im_idx}.jpg')
        to_save = Image.fromarray(im)
        to_save.save(impath)
        im = pool(im)


def extract_and_save_faces(impath, image_annotations, batch_offset):
    original_im = np.array(Image.open(impath).convert("RGB"))
    for face_idx, face_annotation in enumerate(image_annotations):
        save_face(original_im, face_annotation, face_idx + batch_offset)


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
    save_metainfo(image_annotations)
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


def save_metainfo(image_annotations):
    line_idx_to_yfccm_id = {
        item["yfcc100m_line_idx"]: key
        for key, item in fdf_metainfo.items()
    }
    to_save = {

    }
    face_id = 0
    total_faces = sum([len(x) for x in image_annotations])
    validation_size = 50000
    start_validation = total_faces - validation_size

    for image_annotation in image_annotations:
        for face_annotation in image_annotation:
            line_idx = face_annotation["line_idx"]
            yfcc100m_id = line_idx_to_yfccm_id[line_idx]
            face_metainfo = {
                key: item
                for key, item in fdf_metainfo[yfcc100m_id].items()
            }
            new_landmark = face_annotation["landmark"].reshape(2, -1)
            orig_landmark = face_annotation["orig_landmark"]
            assert new_landmark.shape == orig_landmark.shape, f"new_landmark: {new_landmark.shape}, orig_landmark: {orig_landmark.shape}"
            orig_landmark = np.rollaxis(orig_landmark, 1)
            print(orig_landmark.shape)
            face_metainfo["original_bounding_box"] = face_annotation["orig_bbox"].astype(int).tolist()
            face_metainfo["original_landmark"] = orig_landmark.tolist()
            face_metainfo["bounding_box"] = face_annotation["face_bbox"].tolist()
            face_metainfo["landmark"] = face_annotation["landmark"].tolist()
            face_metainfo["yfcc100m_line_idx"] = line_idx

            if face_id >= start_validation:
                face_metainfo["category"] = "validation"
            else:
                face_metainfo["category"] = "training"
            to_save[face_id] = face_metainfo
            face_id += 1

    save_path = os.path.join(TARGET_DIR, "fdf_metainfo.json")
    utils.write_json(to_save, save_path)


def main():
    image_names = get_imnames()
    impaths = []
    image_annotations = []
    with multiprocessing.Pool(1) as pool:
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
    batch_offset = 0
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        for im_idx, annotations in enumerate(image_annotations):
            impath = impaths[im_idx]
            job = pool.apply_async(
                extract_and_save_faces, (impath, annotations, batch_offset)
            )
            batch_offset += len(annotations)
            jobs.append(job)
        for job in tqdm.tqdm(jobs):
            job.get()


if __name__ == "__main__":
    main()
