import numpy as np
import torch
import tqdm
from src.torch_utils import to_cuda, image_to_torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn

model = keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()
to_cuda(model)


def detect_keypoints(img, keypoint_threshold=.3):
    img = image_to_torch(img, cuda=True)[0]
    with torch.no_grad():
        outputs = model([img])
    
    # Shape: [N persons, K keypoints, (x,y,visibility)]
    keypoints = outputs[0]["keypoints"]
    scores = outputs[0]["scores"]
    assert list(scores) == sorted(list(scores))[::-1]
    mask = scores > keypoint_threshold
    keypoints = keypoints[mask, :, :2]
    return keypoints.cpu().numpy()


def batch_detect_keypoints(images, keypoint_threshold=.3):
    images = [image_to_torch(im, cuda=False)[0] for im in images]
    batch_size = 16
    keypoints = []
    scores = []
    if len(images) > 0:
        num_batches = int(np.ceil(len(images) / batch_size))
        with torch.no_grad():
            for i in tqdm.trange(num_batches, desc="Keypoint inference"):
                images_ = images[i*batch_size:(i+1)*batch_size]
                images_ = [to_cuda(_) for _ in images_]
                outputs = model(images_)
                images_ = [_.cpu() for _ in images_]
                keypoints += [o["keypoints"].cpu() for o in outputs]
                scores += [o["scores"].cpu() for o in outputs]
    for i in range(len(scores)):
        im_scores = scores[i]
        im_keypoints = keypoints[i]
        mask = im_scores > keypoint_threshold
        keypoints[i] = im_keypoints[mask, :, :2].numpy()
    return keypoints
