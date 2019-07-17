import multiprocessing
import torch
import numpy as np
import tqdm
import cv2
import os
from src import torch_utils
from .infer import pre_process, post_process, read_args
from ..detection import detection_api
from src.visualization import utils as vis_utils


def anonymize_images(images, im_keypoints, im_bboxes, imsize, generator, batch_size):
  face_info = {}
  face_idx = 0
  anonymized_images = [im.copy() for im in images]
  with multiprocessing.Pool(1) as pool:
    jobs = []
    for im_idx in range(len(images)):
      im = images[im_idx].copy()
      face_keypoints = im_keypoints[im_idx]
      
      face_bboxes = im_bboxes[im_idx]
      for keypoint, bbox in zip(face_keypoints, face_bboxes):
        face_info[face_idx] = {
          "im_idx": im_idx,
          "face_bbox": bbox
        }
        job = pre_process(im.copy(), keypoint, bbox, imsize, False)
        #job = pool.apply_async(pre_process, (im.copy(), keypoint, bbox, imsize, False))
        face_idx += 1
        jobs.append(job)
    for face_idx, job in enumerate(tqdm.tqdm(jobs, desc="Pre-processing images")):
      res = job
      if res is None:
        del face_info[face_idx]
        continue
      torch_input, keypoint, expanded_bbox, new_bbox = res
      face_info[face_idx]["torch_input"] = torch_input
      face_info[face_idx]["translated_keypoint"] = keypoint
      face_info[face_idx]["expanded_bbox"] = expanded_bbox
      #face_info[face_idx]["translated_bbox"] = new_bbox
  if len(face_info) == 0:
    print("WARNING: Did not detect any faces. Returning non-anonymized images")
    return anonymized_images
  face_idx_max = max(len(face_info), max(list(face_info.keys())) + 1)
  for face_idx in range(face_idx_max):
    if face_idx not in face_info:
      for face_idx2 in range(face_idx+1, face_idx_max):
        if face_idx2 in face_info:
          face_info[face_idx2 - 1] = face_info[face_idx2]
          del face_info[face_idx2]
  
  assert len(face_info) == len(face_info.keys())
  torch_faces = torch.empty((len(face_info), 3, imsize, imsize), dtype=torch.float32)
  torch_keypoints = torch.empty((len(face_info), 14), dtype=torch.float32)
  for face_idx in face_info.keys():
    torch_faces[face_idx] = face_info[face_idx]["torch_input"]
    torch_keypoints[face_idx] = face_info[face_idx]["translated_keypoint"]
  
  num_batches = int(np.ceil(len(face_info) / batch_size))
  results = []
  with torch.no_grad():
    for batch_idx in tqdm.trange(num_batches, desc="Anonyimizing faces"):
      im = torch_faces[batch_idx*batch_size:(batch_idx+1)*batch_size]
      keypoints = torch_keypoints[batch_idx*batch_size:(batch_idx+1)*batch_size]
      im = torch_utils.to_cuda(im)
      keypoints = torch_utils.to_cuda(keypoints)
      generated_faces = generator(im, keypoints)
      results.append(generated_faces.cpu())

  generated_faces = torch.cat(results)
  face_idx = 0
  jobs = []
  im_to_face_idx = {}
  for face_idx, info in face_info.items():
    im_idx = info["im_idx"]
    if im_idx not in im_to_face_idx:
      im_to_face_idx[im_idx] = []
    im_to_face_idx[im_idx].append(face_idx)
  
  for im_idx, face_indices in tqdm.tqdm(im_to_face_idx.items(), desc="Post-processing"):
    im = images[im_idx]
    replaced_mask = np.ones_like(im).astype("bool")
    for face_idx in face_indices:
      generated_face = generated_faces[face_idx:face_idx+1]
      expanded_bbox = face_info[face_idx]["expanded_bbox"]
      original_bbox = face_info[face_idx]["face_bbox"]
      im = post_process(im, generated_face, expanded_bbox,
                        original_bbox, replaced_mask)
    anonymized_images[im_idx] = im
  return anonymized_images
  





if __name__ == "__main__":
  generator, imsize, source_path, image_paths, save_path = read_args()
  images = [cv2.imread(p)[:, :, ::-1] for p in image_paths]
  im_bboxes, im_keypoints = detection_api.batch_detect_faces_with_keypoints(images)
  anonymized_images = anonymize_images([i.copy() for i in images],
                                       im_keypoints,
                                       im_bboxes,
                                       imsize,
                                       generator, 128)
  for im_idx in range(len(image_paths)):
    im = images[im_idx]
    anonymized_image = anonymized_images[im_idx]
    filepath = image_paths[im_idx]
    face_boxes, keypoints = im_bboxes[im_idx], im_keypoints[im_idx]
    annotated_im = vis_utils.draw_faces_with_keypoints(im[:, :, ::-1], face_boxes, keypoints)
    to_save = np.concatenate((annotated_im, anonymized_image[:, :, ::-1]), axis=1)

    relative_path = filepath[len(source_path)+1:]
    
    im_savepath = os.path.join(save_path, relative_path)
    print("Saving to:", im_savepath)
    os.makedirs(os.path.dirname(im_savepath), exist_ok=True)
    cv2.imwrite(im_savepath, to_save)
