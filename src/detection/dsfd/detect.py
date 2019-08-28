import torch
import numpy as np
import cv2
from .face_ssd import build_ssd
from .config import resnet152_model_config
from . import torch_utils


class DSFDDetector:

    def __init__(
            self,
            weight_path="dsfd/weights/WIDERFace_DSFD_RES152.pth",
            nms_iou_threshold=.3,
            ):
        self.nms_iou_threshold = nms_iou_threshold
        self.model_loaded = False
        self.weight_path = weight_path

    def load_model(self):
        cfg = resnet152_model_config
        net = build_ssd(cfg)  # initialize SSD

        weight_path = self.weight_path
        net.load_state_dict(torch.load(weight_path,
                                       map_location=torch_utils.get_device()))
        torch_utils.to_cuda(net)
        net.eval()
        print('Finished loading DSFD model!')
        self.net = net
        self.model_loaded = True

    def detect_face(self, image, confidence_threshold, shrink=1.0):
        if not self.model_loaded:
            self.load_model()
        x = image
        if shrink != 1:
            x = cv2.resize(image, None, None, fx=shrink, fy=shrink,
                           interpolation=cv2.INTER_LINEAR)
        height, width = x.shape[:2]
        x = x.astype(np.float32)
        x -= np.array([104, 117, 123], dtype=np.float32)
        x = torch_utils.image_to_torch(x, cuda=True)

        with torch.no_grad():
            y = self.net(x, confidence_threshold, self.nms_iou_threshold)

        detections = y.data.cpu().numpy()

        scale = np.array([width, height, width, height])
        detections[:, :, 1:] *= (scale / shrink)

        # Move axis such that we get #[xmin, ymin, xmax, ymax, det_conf]
        dets = np.roll(detections, 4, axis=-1)
        return dets[0]


