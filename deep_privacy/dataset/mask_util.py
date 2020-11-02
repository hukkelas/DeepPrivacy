import math
import numpy as np
import typing
from PIL import Image, ImageDraw


def random_bbox(img_shape: typing.Tuple[int]) -> typing.Tuple[int]:
    # Numbers taken from https://github.com/JiahuiYu/generative_inpainting/
    # No description given, looks like max height/width of bbox

    img_height, img_width = img_shape
    height_bbox, width_bbox = img_height // 2, img_width // 2
    maxX = img_height - height_bbox
    maxY = img_width - width_bbox
    x0 = int(np.random.uniform(low=0, high=maxX))
    y0 = int(np.random.uniform(low=0, high=maxY))
    return (x0, y0, x0 + width_bbox, y0 + height_bbox)


def get_bbox_mask(img_shape: typing.Tuple[int],
                  fixed_mask: bool) -> np.ndarray:
    img_height, img_width = img_shape
    if fixed_mask:
        assert img_height == img_width
        x0 = img_width // 4
        x1 = x0 + img_width // 2
        bbox = (x0, x0, x1, x1)
    else:
        bbox = random_bbox(img_shape)
    mask = np.ones((img_height, img_width), dtype=bool)
    x0, y0, x1, y1 = bbox
    mask[y0:y1, x0:x1] = 0
    return mask


# Adapted from:
# https://github.com/JiahuiYu/generative_inpainting/blob/master/inpaint_ops.py
# License: Creative Commons Attribution-NonCommercial 4.0 International
def brush_stroke_mask(img_shape: typing.Tuple[int]) -> np.ndarray:
    """Generate mask tensor from bbox.

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """
    img_height, img_width = img_shape
    min_num_vertex = 4
    max_num_vertex = 12
    mean_angle = 2 * math.pi / 5
    angle_range = 2 * math.pi / 15

    # Code was hard-coded to 256x256.
    min_width = 12 / 256 * img_width
    max_width = 40 / 256 * img_height

    def generate_mask(H, W):
        average_radius = math.sqrt(H * H + W * W) / 8
        mask = Image.new('L', (W, H), 0)

        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(
                        2 *
                        math.pi -
                        np.random.uniform(
                            angle_min,
                            angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append(
                (int(
                    np.random.randint(
                        0, w)), int(
                    np.random.randint(
                        0, h))))
            for i in range(num_vertex):
                r = np.random.normal(average_radius, average_radius//2)
                r = np.clip(r, 0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width // 2,
                              v[1] - width // 2,
                              v[0] + width // 2,
                              v[1] + width // 2),
                             fill=1)

        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.bool)
        mask = np.reshape(mask, (H, W))
        return 1 - mask
    return generate_mask(img_shape[0], img_shape[1])


def generate_mask(img_shape: typing.Tuple[int],
                  fixed_mask: bool) -> np.ndarray:
    bbox_mask = get_bbox_mask(img_shape, fixed_mask)
    if fixed_mask:
        return bbox_mask
    brush_mask = brush_stroke_mask(img_shape)
    mask = np.logical_and(bbox_mask, brush_mask)
    return mask.squeeze()
