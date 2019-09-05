import numpy as np
import json


def quadratic_bounding_box(x0, y0, width, height, imshape):
    # We assume that we can create a image that is quadratic without 
    # minimizing any of the sides
    assert width <= min(imshape[:2])
    assert height <= min(imshape[:2])
    min_side = min(height, width)
    if height != width:
        side_diff = abs(height-width)
        # Want to extend the shortest side
        if min_side == height:
            # Vertical side
            height += side_diff
            if height > imshape[0]:
                # Take full frame, and shrink width
                y0 = 0
                height = imshape[0]

                side_diff = abs(height - width)
                width -= side_diff
                x0 += side_diff // 2
            else:
                y0 -= side_diff // 2
                y0 = max(0, y0)
        else:
            # Horizontal side
            width += side_diff
            if width > imshape[1]:
                # Take full frame width, and shrink height
                x0 = 0
                width = imshape[1]

                side_diff = abs(height - width)
                height -= side_diff
                y0 += side_diff // 2
            else:
                x0 -= side_diff // 2
                x0 = max(0, x0)
        # Check that bbox goes outside image
        x1 = x0 + width
        y1 = y0 + height
        if imshape[1] < x1:
            diff = x1 - imshape[1]
            x0 -= diff
        if imshape[0] < y1:
            diff = y1 - imshape[0]
            y0 -= diff
    assert x0 >= 0, "Bounding box outside image."
    assert y0 >= 0, "Bounding box outside image."
    assert x0+width <= imshape[1], "Bounding box outside image."
    assert y0+height <= imshape[0], "Bounding box outside image."
    return x0, y0, width, height


def expand_bounding_box(bbox, percentage, imshape):
    orig_bbox = bbox.copy()
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    x0, y0, width, height = quadratic_bounding_box(x0, y0, width, height, imshape)
    expanding_factor = int(max(height, width) * percentage)

    possible_max_expansion = [(imshape[0] - width)//2,
                              (imshape[1] - height)//2,
                              expanding_factor]

    expanding_factor = min(possible_max_expansion)
    # Expand height

    if expanding_factor > 0:

        y0 = y0 - expanding_factor
        y0 = max(0, y0)

        height += expanding_factor*2
        if height > imshape[0]:
            y0 -= (imshape[0] - height)
            height = imshape[0]

        if height + y0 > imshape[0]:
            y0 -= (height + y0 - imshape[0])

        
        # Expand width
        x0 = x0 - expanding_factor
        x0 = max(0, x0)

        width += expanding_factor*2
        if width > imshape[1]:
            x0 -= (imshape[1] - width)
            width = imshape[1]

        if width + x0 > imshape[1]:
            x0 -= (width + x0 - imshape[1])
    y1 = y0 + height
    x1 = x0 + width
    assert y0 >= 0, "Y0 is minus"
    assert height <= imshape[0], "Height is larger than image."
    assert x0 + width <= imshape[1]
    assert y0 + height <= imshape[0]
    assert width == height, "HEIGHT IS NOT EQUAL WIDTH!!"
    assert x0 >= 0, "Y0 is minus"
    assert width <= imshape[1], "Height is larger than image."
    # Check that original bbox is within new
    x0_o, y0_o, x1_o, y1_o = orig_bbox
    assert x0 <= x0_o, "New bbox is outisde of original. O:{}, N: {}".format(x0_o, x0 )
    assert x1 >= x1_o, "New bbox is outisde of original. O:{}, N: {}".format(x1_o, x1)
    assert y0 <= y0_o, "New bbox is outisde of original. O:{}, N: {}".format(y0_o, y0)
    assert y1 >= y1_o, "New bbox is outisde of original. O:{}, N: {}".format(y1_o, y1)
    #x0, y0, width, height = quadratic_bounding_box(x0, y0, width, height, imshape)
    x0, y0, width, height = [int(_) for _ in [x0, y0, width, height]]
    x1 = x0 + width
    y1 = y0 + height
    return np.array([x0, y0, x1, y1])


def read_json(path):
    print("reading:", path)
    with open(path, "r") as fp:
        return json.load(fp)


def write_json(obj, path):
    with open(path, "w") as fp:
        json.dump(obj, fp)


def is_keypoint_within_bbox(x0, y0, x1, y1, keypoint):
    keypoint = keypoint[:, :3] # only nose + eyes are relevant
    kp_X = keypoint[0, :]
    kp_Y = keypoint[1, :]
    within_X = np.all(kp_X >= x0) and np.all(kp_X <= x1)
    within_Y = np.all(kp_Y >= y0) and np.all(kp_Y <= y1)
    return within_X and within_Y


def expand_bbox_simple(bbox, percentage):
    x0, y0, x1, y1 = bbox.astype(float)
    width = x1 - x0
    height = y1 - y0
    x_c = int(x0) + width//2
    y_c = int(y0) + height//2
    avg_size = max(width, height)
    new_width = avg_size * (1 + percentage)
    x0 = x_c - new_width//2
    y0 = y_c - new_width//2
    x1 = x_c + new_width//2
    y1 = y_c + new_width//2
    return np.array([x0, y0, x1, y1]).astype(int)


def pad_image(im, bbox):
    x0, y0, x1, y1 = bbox
    if x0 < 0:
        pad_im = np.zeros((im.shape[0], abs(x0), im.shape[2]), dtype=np.uint8)
        im = np.concatenate((pad_im, im), axis=1)
        x1 += abs(x0)
        x0 = 0
    if y0 < 0:
        pad_im = np.zeros((abs(y0), im.shape[1], im.shape[2]), dtype=np.uint8)
        im = np.concatenate((pad_im, im), axis=0)
        y1 += abs(y0)
        y0 = 0
    if x1 >= im.shape[1]:
        pad_im = np.zeros((im.shape[0], x1 - im.shape[1] + 1, im.shape[2]), dtype=np.uint8)
        im = np.concatenate((im, pad_im), axis=1)
    if y1 >= im.shape[0]:
        pad_im = np.zeros((y1 - im.shape[0] + 1, im.shape[1], im.shape[2]), dtype=np.uint8)
        im = np.concatenate((im, pad_im), axis=0)
    return im[y0:y1, x0:x1]


def cut_face(im, bbox, simple_expand):
    if simple_expand or (bbox < 0).any() or (bbox[2] > im.shape[1]) or (bbox[3] > im.shape[0]):
        return pad_image(im, bbox)
    x0, y0, x1, y1 = bbox
    return im[y0:y1, x0:x1]



def expand_bbox(bbox, imshape, simple_expand, default_to_simple=False, expansion_factor1=0.35):
    assert bbox.shape == (4,), f"BBox shape was: {bbox.shape}"
    bbox = bbox.astype(float)
    if simple_expand:
        return expand_bbox_simple(bbox, 0.4)
    try:
        return expand_bounding_box(bbox, expansion_factor1, imshape)
    except AssertionError:
        return expand_bbox_simple(bbox, expansion_factor1*2)
