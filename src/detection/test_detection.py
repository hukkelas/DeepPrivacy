from .detection_api import detect_faces_with_keypoints
import matplotlib.pyplot as plt 
import os
from src.visualization import utils as vis_utils
import cv2


os.makedirs("test_examples/out", exist_ok=True)

impath = "test_examples/maxresdefault.jpg"

im = plt.imread(impath)

face_bboxes, keypoints = detect_faces_with_keypoints(im, keypoint_threshold=0.1)

im = im[:, :, ::-1]
im = vis_utils.draw_faces_with_keypoints(im, face_bboxes, keypoints)

cv2.imwrite("test_examples/out/test.jpg", im)