import imagehash
import json
import numpy as np
import torch
import os
from PIL import Image
from src.torch_utils import to_cuda
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from apex import amp

model = keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()
to_cuda(model)

image_hash_file = os.path.join(".cache/detection/keypoint_detections.json")
os.makedirs(os.path.dirname(image_hash_file), exist_ok=True)
image_hash_to_detections = {

}
if os.path.isfile(image_hash_file):
  with open(image_hash_file, "r") as fp:
    image_hash_to_detections = json.load(fp)

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
  orig_images = images
  detections = [get_saved_detection(im) for im in images]
  images = [np.moveaxis(im, 2, 0) / im.max() for im, det in zip(images, detections) if det is None]

  images = [to_cuda(torch.from_numpy(im).float()) for im in images]
  outputs = []
  if len(images) > 0:
    with torch.no_grad():
      outputs = model(images)
  keypoints = [o["keypoints"] for o in outputs]
  scores = [o["scores"] for o in outputs]
  for i in range(len(scores)):
    im_scores = scores[i]
    im_keypoints = keypoints[i]
    mask = im_scores > keypoint_threshold
    keypoints[i] = im_keypoints[mask, :, :2].cpu().numpy()
  idx = 0
  for d_idx in range(len(detections)):
    if detections[d_idx] is None:
      detections[d_idx] = keypoints[idx]
      save_detection(orig_images[d_idx], keypoints[idx])
      idx += 1
  write_detections_to_file()
  return detections

def get_saved_detection(im):
  im = Image.fromarray(im)
  hash_id = str(imagehash.average_hash(im))
  if hash_id in image_hash_to_detections:
    return np.array(image_hash_to_detections[hash_id])
  return None

def save_detection(im, detection):
  im = Image.fromarray(im)
  hash_id = imagehash.average_hash(im)
  image_hash_to_detections[str(hash_id)] = detection.tolist()

def write_detections_to_file():
  with open(image_hash_file, "w") as fp:
    json.dump(image_hash_to_detections, fp)
