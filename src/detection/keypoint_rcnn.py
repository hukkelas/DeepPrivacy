import imagehash
import json
import numpy as np
import torch
import tqdm
import os
from PIL import Image
from src.torch_utils import to_cuda, image_to_torch
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
  orig_images = images
  detections = [get_saved_detection(im, keypoint_threshold) for im in images]
  images = [image_to_torch(im, cuda=False)[0] for im, det in zip(images, detections) if det is None]
  batch_size = 32
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
  idx = 0
  for d_idx in range(len(detections)):
    if detections[d_idx] is None:
      detections[d_idx] = keypoints[idx]
      save_detection(orig_images[d_idx], keypoints[idx], keypoint_threshold)
      idx += 1
  write_detections_to_file()
  return detections

def get_saved_detection(im, keypoint_threshold):
  im = Image.fromarray(im)
  hash_id = str(imagehash.average_hash(im))
  if hash_id in image_hash_to_detections:
    if str(keypoint_threshold) in image_hash_to_detections[hash_id]:
      dets = np.array(image_hash_to_detections[hash_id][str(keypoint_threshold)])
      if len(dets) == 0:
        return np.empty((0, 17, 2))
      return dets
  return None

def save_detection(im, detection, keypoint_threshold):
  im = Image.fromarray(im)
  hash_id = imagehash.average_hash(im)
  image_hash_to_detections[str(hash_id)] = {
    keypoint_threshold: detection.tolist()
  }

def write_detections_to_file():
  with open(image_hash_file, "w") as fp:
    json.dump(image_hash_to_detections, fp)
