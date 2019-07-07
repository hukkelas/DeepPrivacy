import cv2
import matplotlib

colors = list(matplotlib.colors.cnames.values())
hex_to_rgb = lambda h: tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
colors = [hex_to_rgb(x[1:]) for x in colors]


def draw_faces_with_keypoints(im, bboxes, keypoints):
    radius = max(int(max(im.shape)*0.0025), 1)
    for c_idx, (bbox, keypoint) in enumerate(zip(bboxes, keypoints)):
        color = colors[c_idx % len(colors)]
        x0, y0, x1, y1 = bbox
        im = cv2.rectangle(im, (x0, y0), (x1, y1), color)
        for (x,y) in keypoint:
            im = cv2.circle(im, (x,y), radius, color)
    return im