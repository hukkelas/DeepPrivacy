#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from detectron.utils.vis import convert_from_cls_format

import numpy as np
import pycocotools.mask as mask_util
import detectron.utils.env as envu

# Matplotlib requires certain adjustments in some environments
# Must happen before importing matplotlib
envu.set_up_matplotlib()
import matplotlib.pyplot as plt

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--always-out',
        dest='out_when_no_box',
        help='output image even when no object is found',
        action='store_true'
    )
    parser.add_argument(
        '--output-ext',
        dest='output_ext',
        help='output image file format (default: pdf)',
        default='pdf',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--kp-thresh',
        dest='kp_thresh',
        help='Threshold for visualizing keypoints',
        default=2.0,
        type=float
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


CFG_PATH = "/detectron/configs/12_2017_baselines/e2e_keypoint_rcnn_X-101-64x4d-FPN_s1x.yaml"
merge_cfg_from_file(CFG_PATH)
WEIGHT_PATH = "https://dl.fbaipublicfiles.com/detectron/37732415/12_2017_baselines/e2e_keypoint_rcnn_X-101-64x4d-FPN_s1x.yaml.16_57_48.Spqtq3Sf/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
cfg.NUM_GPUS = 1
weights = cache_url(WEIGHT_PATH, cfg.DOWNLOAD_CACHE)

assert_and_infer_cfg(cache_urls=False)
assert not cfg.MODEL.RPN_ONLY,  'RPN models are not supported'
assert not cfg.TEST.PRECOMPUTED_PROPOSALS, 'Models that require precomputed proposals are not supported'

model = infer_engine.initialize_model_from_cfg(weights)

def predict_keypoint(impath, kp_thresh=0.3):
    im = cv2.imread(impath)
    max_res = 1080
    if max(im.shape) > max_res:
        scale_factor =  max_res / max(im.shape)
        im = cv2.resize(im, (0,0), fx=scale_factor, fy=scale_factor)
        
    else:
        scale_factor = 1
    with c2_utils.NamedCudaScope(0):
        print(im.shape)
        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
            model, im, None
        )

    keypoints = extract_keypoints(im, "lol", cls_boxes, cls_segms, cls_keyps, thresh=kp_thresh)
    if keypoints is None: return []
    keypoints = np.array(keypoints)
    keypoints /= scale_factor
    # Try to sort...
    score = keypoints[:, 2, :].sum(axis=1)
    sortex_idx = np.argsort(score)[::-1]
    keypoints = keypoints[sortex_idx]
    return keypoints


from matplotlib.patches import Polygon

"""
keypoint indices: 
'nose',
'right_eye',
'left_eye',
'right_ear',
'left_ear',
'right_shoulder',
'left_shoulder',
"""
def extract_keypoints(
        im, output_dir, boxes, segms=None, keypoints=None, thresh=0.3,
        kp_thresh=2, dpi=200, box_alpha=0.0, dataset=None, show_class=False,
        ext='pdf', out_when_no_box=False):

    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)

    if (boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh) and not out_when_no_box:
        return

    if boxes is None:
        sorted_inds = [] # avoid crash when 'boxes' is None
    else:
        # Display in largest to smallest order to reduce occlusion
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)
    keypoints_to_return = []
    for i in sorted_inds:
        score = boxes[i, -1]
        if score < thresh:
            continue


        if keypoints is not None and len(keypoints) > i:
            kps = keypoints[i]
            kps = kps[:, :7]
            if (kps[2, :] <= kp_thresh).sum() == 7:
                continue
            keypoints_to_return.append(kps)
    return keypoints_to_return

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
