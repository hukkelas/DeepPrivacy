import numpy as np
import tqdm
from .SFD_pytorch.wider_eval_pytorch import detect_and_supress
from . import keypoint_rcnn
from .utils import match_bbox_keypoint



def detect_faces_with_keypoints(img, face_threshold=0.5, keypoint_threshold=0.3):
  face_bboxes = detect_and_supress(img[:, :, ::-1], face_threshold)
  keypoints = keypoint_rcnn.detect_keypoints(img, keypoint_threshold)[:, :7, :]
  face_bboxes, keypoints = match_bbox_keypoint(face_bboxes, keypoints)
  return face_bboxes, keypoints

def batch_detect_faces_with_keypoints(images, face_threshold=.5, keypoint_threshold=.3):
  im_bboxes = []
  for im in tqdm.tqdm(images, desc="Batch detecting faces"):
    im_bboxes.append(
      detect_and_supress(im[:, :, ::-1], face_threshold)
    )
  
  keypoints = keypoint_rcnn.batch_detect_keypoints(images, keypoint_threshold)
  for i in range(len(im_bboxes)):
    face_bboxes = im_bboxes[i]
    face_kps = keypoints[i][:, :7, :]
    face_bboxes, face_kps = match_bbox_keypoint(face_bboxes, face_kps)
    im_bboxes[i] = face_bboxes
    keypoints[i] = face_kps
  return im_bboxes, keypoints
    
  
