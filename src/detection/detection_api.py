import numpy as np
import tqdm
import imagehash
import json
import os
from PIL import Image
from .dsfd.detect import get_face_detections
from .SFD_pytorch.wider_eval_pytorch import detect_and_supress
from . import keypoint_rcnn
from .utils import match_bbox_keypoint

image_hash_file = os.path.join(".cache/detection/face_detections.json")
os.makedirs(os.path.dirname(image_hash_file), exist_ok=True)
image_hash_to_detections = {

}
if os.path.isfile(image_hash_file):
  with open(image_hash_file, "r") as fp:
    image_hash_to_detections = json.load(fp)


def clip_detections(detections, imshape):
    detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, imshape[1])
    detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, imshape[0])
    return detections



def batch_detect_faces(images, face_threshold=0.5):
    im_bboxes = []
    for im in tqdm.tqdm(images, desc="Batch detecting faces"):
        det = get_face_detections(im[:, :, ::-1])
        det = det[:, :4].astype(int)
        im_bboxes.append(det)
    im_bboxes = [clip_detections(dets, im.shape) 
                 for dets, im in zip(im_bboxes, images)]
    return im_bboxes
      
  
def detect_faces_with_keypoints(img, face_threshold=0.5, keypoint_threshold=0.3):
    face_bboxes = detect_and_supress(img[:, :, ::-1], face_threshold)
    keypoints = keypoint_rcnn.detect_keypoints(img, keypoint_threshold)[:, :7, :]
    face_bboxes, keypoints = match_bbox_keypoint(face_bboxes, keypoints)
    return face_bboxes, keypoints


def batch_detect_faces_with_keypoints(images, face_threshold=.5, keypoint_threshold=.3, im_bboxes=None):
    if im_bboxes is None:
        im_bboxes = batch_detect_faces(images, face_threshold)
    
    keypoints = keypoint_rcnn.batch_detect_keypoints(images, keypoint_threshold)
    for i in range(len(im_bboxes)):
        face_bboxes = im_bboxes[i]
        face_kps = keypoints[i][:, :7, :]
        face_bboxes, face_kps = match_bbox_keypoint(face_bboxes, face_kps)
        im_bboxes[i] = face_bboxes
        keypoints[i] = face_kps
    return im_bboxes, keypoints
