import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


colors = list(matplotlib.colors.cnames.values())


def hex_to_rgb(h): return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


colors = [hex_to_rgb(x[1:]) for x in colors]
colors = [(255, 0, 0)] + colors


def draw_faces_with_keypoints(
        im,
        im_bboxes,
        im_keypoints,
        radius=None,
        black_out_face=False,
        color_override=None
):
    im = im.copy()
    if im_keypoints is None:
        assert im_bboxes is not None, "Image bboxes cannot be None."
        im_keypoints = [None for i in range(len(im_bboxes))]
    if im_bboxes is None:
        im_bboxes = [None for i in range(len(im_keypoints))]
    if radius is None:
        radius = max(int(max(im.shape) * 0.0025), 1)
    for c_idx, (bbox, keypoint) in enumerate(zip(im_bboxes, im_keypoints)):
        color = color_override
        if color_override is None:
            color = colors[c_idx % len(colors)]

        if bbox is not None:
            x0, y0, x1, y1 = bbox
            if black_out_face:
                im[y0:y1, x0:x1, :] = 0
            else:
                im = cv2.rectangle(im, (x0, y0), (x1, y1), color)
        if keypoint is None:
            continue
        for x, y in keypoint:
            im = cv2.circle(im, (int(x), int(y)), radius, color)
    if not isinstance(im, np.ndarray):
        return im.get()
    return im


def np_make_image_grid(images, nrow, pad=2):
    height, width = images[0].shape[:2]
    ncol = int(np.ceil(len(images) / nrow))
    im_result = np.zeros((nrow * (height + pad), ncol * (width + pad), 3),
                         dtype=images[0].dtype)
    im_idx = 0
    for row in range(nrow):
        for col in range(ncol):
            if im_idx == len(images):
                break
            im = images[im_idx]
            im_idx += 1
            im_result[row * (pad + height): (row) * (pad + height) + height,
                      col * (pad + width): (col) * (pad + width) + width, :] = im
    return im_result


def add_text(im, x, y, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (x, y + 10)
    fontScale = .4
    fontColor = (255, 255, 255)
    backgroundColor = (0, 0, 0)
    lineType = 1

    cv2.putText(im, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                backgroundColor,
                lineType * 2)
    cv2.putText(im, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)


def add_label_y(im, positions, labels):
    # positions [(x, y)]
    im = im.copy()
    assert len(positions) == len(labels)
    for pos, label in zip(positions, labels):
        add_text(im, 0, pos, label)
    return im


def plot_bbox(bbox):
    x0, y0, x1, y1 = bbox
    plt.plot([x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0])


def pad_im_as(im, target_im):
    assert len(im.shape) == 3
    assert len(target_im.shape) == 3
    assert im.shape[0] <= target_im.shape[0]
    assert im.shape[1] <= target_im.shape[1],\
        f"{im.shape}, {target_im.shape}"
    pad_h = abs(im.shape[0] - target_im.shape[0]) // 2
    pad_w = abs(im.shape[1] - target_im.shape[1]) // 2
    im = np.pad(im, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
    assert im.shape == target_im.shape
    return im
