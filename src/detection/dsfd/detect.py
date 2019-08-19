import torch
from .face_ssd import build_ssd
from .config import resnet152_model_config
import numpy as np
import cv2
import time
import os

if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    #torch.backends.cudnn.benchmark = True

confidence_threshold = .1
nms_iou_threshold = .3
use_multiscale_detect = False
use_flip_detect = False
use_image_pyramid_detect = False

def detect_face(image, shrink):
    t = time.time()
    x = image
    if shrink != 1:
        x = cv2.resize(image, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)

    width = x.shape[1]
    height = x.shape[0]
    x = x.astype(np.float32)
    x -= np.array([104, 117, 123],dtype=np.float32)

    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)

    x = x.cuda()
    #net.priorbox = PriorBoxLayer(width,height)
    with torch.no_grad():
        y = net(x, confidence_threshold, nms_iou_threshold)
    detections = y.data.cpu().numpy()
    scale = np.array([width, height, width, height])

    boxes = []
    scores = []
    for i in range(detections.shape[1]):
        score = detections[0, i, 0]
        pt = (detections[0, i, 1:]*scale)
        boxes.append([pt[0],pt[1],pt[2],pt[3]])
        scores.append(score)
    det_conf = np.array(scores)
    boxes = np.array(boxes)

    if boxes.shape[0] == 0:
        return np.array([[0,0,0,0,0.001]])

    det_xmin = boxes[:,0] / shrink
    det_ymin = boxes[:,1] / shrink
    det_xmax = boxes[:,2] / shrink
    det_ymax = boxes[:,3] / shrink
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

    keep_index = np.where(det[:, 4] >= 0)[0]
    det = det[keep_index, :]
    return det


def multi_scale_test(image, max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(image, st)
    if max_im_shrink > 0.75:
        det_s = np.row_stack((det_s,detect_face(image,0.75)))
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]
    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = detect_face(image, bt)

    # enlarge small iamge x times for small face
    if max_im_shrink > 1.5:
        det_b = np.row_stack((det_b,detect_face(image,1.5)))
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink: # and bt <= 2:
            det_b = np.row_stack((det_b, detect_face(image, bt)))
            bt *= 2

        det_b = np.row_stack((det_b, detect_face(image, max_im_shrink)))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b


def multi_scale_test_pyramid(image, max_shrink):
    # shrink detecting and shrink only detect big face
    det_b = detect_face(image, 0.25)
    index = np.where(
        np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1)
        > 30)[0]
    det_b = det_b[index, :]

    st = [1.25, 1.75, 2.25]
    for i in range(len(st)):
        if (st[i] <= max_shrink):
            det_temp = detect_face(image, st[i])
            # enlarge only detect small face
            if st[i] > 1:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) < 100)[0]
                det_temp = det_temp[index, :]
            else:
                index = np.where(
                    np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) > 30)[0]
                det_temp = det_temp[index, :]
            det_b = np.row_stack((det_b, det_temp))
    return det_b


def flip_test(image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    if det.shape[0] == 0:
        return det[0:750, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum
    try:
        dets = dets[0:750, :]
    except:
        dets = det[0:750, :]
    return dets


# load net
cfg = resnet152_model_config
net = build_ssd(cfg)  # initialize SSD

weight_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.path.join("weights", "WIDERFace_DSFD_RES152.pth")
)
net.load_state_dict(torch.load(weight_path))
net.cuda()
net.eval()
print('Finished loading model!')

# load data


def get_face_detections(image):
    t = time.time()
    max_im_shrink = (0x7fffffff / 200.0 / (image.shape[0] * image.shape[1])) ** 0.5 # the max size of input image for caffe
    max_im_shrink = 3 if max_im_shrink > 3 else max_im_shrink
    shrink = max_im_shrink if max_im_shrink < 1 else 1
    dets = []
    det0 = detect_face(image, shrink)  # origin test
    dets.append(det0)
    if use_flip_detect:
        det1 = flip_test(image, shrink)    # flip test
        dets.append(det1)
    if use_multiscale_detect:
        det2, det3 = multi_scale_test(image, max_im_shrink)
        dets.extend([det2, det3])
    if use_image_pyramid_detect:
        det4 = multi_scale_test_pyramid(image, max_im_shrink)
        dets.append(det4)
    if len(dets) > 1:
        dets = np.row_stack((det0, det1, det2, det3, det4))
        dets = bbox_vote(dets)
    else:
        dets = dets[0]
    return dets

