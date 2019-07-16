from torchvision.models.detection import keypointrcnn_resnet50_fpn
import numpy as np
import torch
from src.torch_utils import to_cuda
from apex import amp

model = keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()
to_cuda(model)

def detect_keypoints(img, keypoint_threshold=.3):
  img = np.rollaxis(img, 2) 
  img = img / img.max()
  img = torch.from_numpy(img).float()
  img = to_cuda(img)
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
  images = [np.moveaxis(im, 2, 0) / im.max() for im in images]
  images = [to_cuda(torch.from_numpy(im).float()) for im in images]
  with torch.no_grad():
    outputs = model(images)
  keypoints = [o["keypoints"] for o in outputs]
  scores = [o["scores"] for o in outputs]
  for i in range(len(scores)):
    im_scores = scores[i]
    im_keypoints = keypoints[i]
    mask = im_scores > keypoint_threshold
    #print(mask)
    keypoints[i] = im_keypoints[mask, :, :2].cpu().numpy()
  return keypoints