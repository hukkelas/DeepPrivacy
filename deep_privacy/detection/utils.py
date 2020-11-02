import numpy as np


def is_keypoint_within_bbox(x0, y0, x1, y1, keypoint):
    keypoint = keypoint[:3, :]  # only nose + eyes are relevant
    kp_X = keypoint[:, 0]
    kp_Y = keypoint[:, 1]
    within_X = np.all(kp_X >= x0) and np.all(kp_X <= x1)
    within_Y = np.all(kp_Y >= y0) and np.all(kp_Y <= y1)
    return within_X and within_Y


def match_bbox_keypoint(bounding_boxes, keypoints):
    """
        bounding_boxes shape: [N, 5]
        keypoints: [N persons, K keypoints, (x, y)]
    """
    if len(bounding_boxes) == 0 or len(keypoints) == 0:
        return np.empty((0, 5)), np.empty((0, 7, 2))
    assert bounding_boxes.shape[1] == 4,\
        f"Shape was : {bounding_boxes.shape}"
    assert keypoints.shape[-1] == 2,\
        f"Expected (x,y) in last axis, got: {keypoints.shape}"
    assert keypoints.shape[1] in (5, 7),\
        f"Expeted 5 or 7 keypoints. Keypoint shape was: {keypoints.shape}"

    matches = []
    for bbox_idx, bbox in enumerate(bounding_boxes):
        keypoint = None
        for kp_idx, keypoint in enumerate(keypoints):
            if kp_idx in (x[1] for x in matches):
                continue
            if is_keypoint_within_bbox(*bbox, keypoint):
                matches.append((bbox_idx, kp_idx))
                break
    keypoint_idx = [x[1] for x in matches]
    bbox_idx = [x[0] for x in matches]
    return bounding_boxes[bbox_idx], keypoints[keypoint_idx]
