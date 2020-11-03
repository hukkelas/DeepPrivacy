import numpy as np
import torch
from deep_privacy.torch_utils import to_cuda, image_to_torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn


class RCNNKeypointDetector:

    def __init__(self, keypoint_treshold: float, batch_size: int):
        super().__init__()
        model = keypointrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        to_cuda(model)
        self.batch_size = batch_size
        self.keypoint_threshold = keypoint_treshold
        self.model = model

    def detect_keypoints(self, img):
        img = image_to_torch(img, cuda=True)[0]
        with torch.no_grad():
            outputs = self.model([img])

        # Shape: [N persons, K keypoints, (x,y,visibility)]
        keypoints = outputs[0]["keypoints"]
        scores = outputs[0]["scores"]
        assert list(scores) == sorted(list(scores))[::-1]
        mask = scores >= self.keypoint_threshold
        keypoints = keypoints[mask, :, :2]
        return keypoints.cpu().numpy()

    def batch_detect_keypoints(self, images):
        images = [image_to_torch(im, cuda=False)[0] for im in images]
        keypoints = []
        if len(images) > 0:
            num_batches = int(np.ceil(len(images) / self.batch_size))
            with torch.no_grad():
                for i in range(num_batches):
                    images_ = images[i * self.batch_size:(i + 1) * self.batch_size]
                    images_ = [to_cuda(_) for _ in images_]
                    outputs = self.model(images_)
                    images_ = [_.cpu() for _ in images_]
                    kps = [o["keypoints"].cpu() for o in outputs]
                    score = [o["scores"].cpu() for o in outputs]
                    masks = [imscore >= self.keypoint_threshold
                             for imscore in score]
                    kps = [kp[mask, :, :2].numpy()
                           for kp, mask in zip(kps, masks)]
                    keypoints += kps
        return keypoints
