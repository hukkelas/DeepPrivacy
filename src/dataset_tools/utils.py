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


def expand_bounding_box(x0, y0, x1, y1, percentage, imshape):
    x0_o = x0
    y0_o = y0
    x1_o = x1
    y1_o = y1
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

    assert x0 <= x0_o, "New bbox is outisde of original. O:{}, N: {}".format(x0_o, x0 )
    assert x1 >= x1_o, "New bbox is outisde of original. O:{}, N: {}".format(x1_o, x1)
    assert y0 <= y0_o, "New bbox is outisde of original. O:{}, N: {}".format(y0_o, y0)
    assert y1 >= y1_o, "New bbox is outisde of original. O:{}, N: {}".format(y1_o, y1)
    #x0, y0, width, height = quadratic_bounding_box(x0, y0, width, height, imshape)
    return x0, y0, width, height

def read_json(path):
    with open(path, "r") as fp:
        return json.load(fp)


def is_keypoint_within_bbox(x0, y0, x1, y1, keypoint):
    keypoint = keypoint[:, :3] # only nose + eyes are relevant
    kp_X = keypoint[0, :]
    kp_Y = keypoint[1, :]
    within_X = np.all(kp_X >= x0) and np.all(kp_X <= x1)
    within_Y = np.all(kp_Y >= y0) and np.all(kp_Y <= y1)
    return within_X and within_Y
