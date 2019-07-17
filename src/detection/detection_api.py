import numpy as np
import tqdm
import imagehash
import json
import os
from PIL import Image
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

def detect_faces_with_keypoints(img, face_threshold=0.5, keypoint_threshold=0.3):
  face_bboxes = detect_and_supress(img[:, :, ::-1], face_threshold)
  keypoints = keypoint_rcnn.detect_keypoints(img, keypoint_threshold)[:, :7, :]
  face_bboxes, keypoints = match_bbox_keypoint(face_bboxes, keypoints)
  return face_bboxes, keypoints

def batch_detect_faces_with_keypoints(images, face_threshold=.5, keypoint_threshold=.3, im_bboxes=None):
  if im_bboxes is None:
    im_bboxes = []
    for im in tqdm.tqdm(images, desc="Batch detecting faces"):
      det = get_saved_detection(im)
      if det is not None:
        im_bboxes.append(det)
        continue
      im_bboxes.append(
        detect_and_supress(im[:, :, ::-1], face_threshold)
      )
      save_detection(im, im_bboxes[-1])
  
  keypoints = keypoint_rcnn.batch_detect_keypoints(images, keypoint_threshold)
  for i in range(len(im_bboxes)):
    face_bboxes = im_bboxes[i]
    face_kps = keypoints[i][:, :7, :]
    face_bboxes, face_kps = match_bbox_keypoint(face_bboxes, face_kps)
    im_bboxes[i] = face_bboxes
    keypoints[i] = face_kps
  write_detections_to_file()
  return im_bboxes, keypoints

def get_saved_detection(im):
  im = Image.fromarray(im)
  hash_id = str(imagehash.average_hash(im))
  if hash_id in image_hash_to_detections:
    return np.array(image_hash_to_detections[hash_id])
  return None

def save_detection(im, detection):
  im = Image.fromarray(im)
  hash_id = imagehash.average_hash(im)
  image_hash_to_detections[str(hash_id)] = detection.tolist()

def write_detections_to_file():
  with open(image_hash_file, "w") as fp:
    json.dump(image_hash_to_detections, fp)
