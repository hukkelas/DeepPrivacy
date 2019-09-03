import cv2
import matplotlib
import numpy as np


colors = list(matplotlib.colors.cnames.values())


def hex_to_rgb(h): return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


colors = [hex_to_rgb(x[1:]) for x in colors]
colors = [(255, 0, 0)] + colors


def draw_faces_with_keypoints(
        im,
        im_bboxes,
        im_keypoints,
        radius=None,
        black_out_face=False
        ):
    im = im.copy()
    if im_keypoints is None:
        assert im_bboxes is not None, "Image bboxes cannot be None."
        im_keypoints = [None for i in range(len(im_bboxes))]
    if im_bboxes is None:
        im_bboxes = [None for i in range(len(im_keypoints))]
    if radius is None:
        radius = max(int(max(im.shape)*0.0025), 1)
    for c_idx, (bbox, keypoint) in enumerate(zip(im_bboxes, im_keypoints)):
        color = colors[c_idx % len(colors)]
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            if black_out_face:
                im[y0:y1, x0:x1, :] = 128
            else:
                im = cv2.rectangle(im, (x0, y0), (x1, y1), color)
        if keypoint is None:
            continue
        for x, y in keypoint:
            im = cv2.circle(im, (int(x), int(y)), radius, color)
    if type(im) != np.ndarray:
        return im.get()
    return im


def draw_faces(im, bboxes):
    for c_idx, bbox in enumerate(bboxes):
        color = colors[c_idx % len(colors)]
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        im = cv2.rectangle(im, (x0, y0), (x1, y1), color)
    if type(im) != np.ndarray:
        return im.get()
    return im


def np_make_image_grid(images, nrow, pad=2):
    imsize = images[0].shape[0]
    ncol = int(np.ceil(len(images) / 2))
    im_result = np.zeros((nrow*(imsize+pad), ncol*(imsize+pad), 3),
                         dtype=images[0].dtype)
    im_idx = 0
    for row in range(nrow):
        for col in range(ncol):
            if im_idx == len(images):
                break
            im = images[im_idx]
            im_idx += 1
            im_result[row*(pad+imsize): (row)*(pad+imsize) + imsize,
                      col*(pad+imsize): (col)*(pad+imsize) + imsize, :] = im
    return im_result
