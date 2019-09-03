def is_height_larger(bbox, image_shape, max_face_size):
    x0, y0, x1, y1 = bbox
    face_size = (y1 - y0) / image_shape[0]
    return face_size >= max_face_size


def is_width_larger(bbox, image_shape, max_face_size):
    x0, y0, x1, y1 = bbox
    face_size = (x1 - x0) / image_shape[1]
    return face_size >= max_face_size


def filter_bboxes(bboxes, imshape, max_face_size, keypoints=None, filter_type="height"):
    keep_idx = []
    filter_func = is_height_larger if filter_type == "height" else is_width_larger
    for idx, bbox in enumerate(bboxes):
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        if not filter_func(bbox, imshape, max_face_size):
            keep_idx.append(idx)
    if keypoints is not None:
        return bboxes[keep_idx], keypoints[keep_idx]
    return bboxes[keep_idx]


def filter_image_bboxes(im_bboxes, im_keypoints, imshapes, max_face_size, filter_type):
    new_boxes = []
    new_keypoints = []
    for im_idx in range(len(im_bboxes)):
        bboxes = im_bboxes[im_idx]
        keypoints = im_keypoints[im_idx]
        imshape = imshapes[im_idx]
        bboxes, keypoints = filter_bboxes(bboxes, imshape, max_face_size,
                                          keypoints, filter_type)
        new_boxes.append(bboxes)
        new_keypoints.append(keypoints)
    return new_boxes, new_keypoints

