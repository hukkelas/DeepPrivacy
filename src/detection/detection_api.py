from .SFD_pytorch.wider_eval_pytorch import detect_and_supress
from .keypoint_rcnn import detect_keypoints
from .utils import match_bbox_keypoint



def detect_faces_with_keypoints(img, face_threshold=0.5, keypoint_threshold=0.3):
    face_bboxes = detect_and_supress(img, face_threshold)
    keypoints = detect_keypoints(img, keypoint_threshold)[:, :7, :]
    face_bboxes, keypoints = match_bbox_keypoint(face_bboxes, keypoints)
    return face_bboxes, keypoints
