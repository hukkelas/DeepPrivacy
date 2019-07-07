import numpy as np
import torch


def nms(dets, thresh):
    if 0==len(dets): return []
    x1,y1,x2,y2,scores = dets[:, 0],dets[:, 1],dets[:, 2],dets[:, 3],dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1,yy1 = np.maximum(x1[i], x1[order[1:]]),np.maximum(y1[i], y1[order[1:]])
        xx2,yy2 = np.minimum(x2[i], x2[order[1:]]),np.minimum(y2[i], y2[order[1:]])

        w,h = np.maximum(0.0, xx2 - xx1 + 1),np.maximum(0.0, yy2 - yy1 + 1)
        ovr = w*h / (areas[i] + areas[order[1:]] - w*h)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes