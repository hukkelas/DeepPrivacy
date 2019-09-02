import numpy as np
import tqdm
from .dsfd.detect import DSFDDetector
from . import keypoint_rcnn
from .utils import match_bbox_keypoint

face_detector = DSFDDetector("src/detection/dsfd/weights/WIDERFace_DSFD_RES152.pth")


def clip_detections(detections, imshape):
    detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, imshape[1])
    detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, imshape[0])
    height = detections[:, 3] - detections[:, 1]
    width = detections[:, 2] - detections[:, 0]
    detections = detections[width > 0]
    detections = detections[height > 0]
    return detections


def batch_detect_faces(images, face_threshold=0.5):
    im_bboxes = []
    for im in tqdm.tqdm(images, desc="Batch detecting faces"):
        det = face_detector.detect_face(im[:, :, ::-1], face_threshold)
        det = det[:, :4].astype(int)
        im_bboxes.append(det)
    im_bboxes = [clip_detections(dets, im.shape) 
                 for dets, im in zip(im_bboxes, images)]
    return im_bboxes


def detect_faces_with_keypoints(img, face_threshold=0.5, keypoint_threshold=0.3):
    face_bboxes = face_detector.detect_face(img[:, :, ::-1], face_threshold)
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
