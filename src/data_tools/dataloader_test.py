import torch
from .dataloaders_v2 import load_dataset
from ..visualization import utils as vis_utils
from .. import torch_utils
import matplotlib.pyplot as plt
from . import data_utils

def keypoint1d_to_2d(keypoints):
  assert len(keypoints.shape) == 1
  x = keypoints[range(0, len(keypoints), 2)]
  y = keypoints[range(1, len(keypoints), 2)]
  kp = torch.stack((x,y), dim=1)
  return kp

dl_train, dl_val = load_dataset("yfcc100m128", 
                                batch_size=128,
                                imsize=128,
                                full_validation=False,
                                pose_size=14,
                                load_fraction=True)

num_ims = 100

dl = dl_train

dl.update_next_transition_variable(1.0)
next(iter(dl))
to_save_condition = []
to_save_ims = []
for ims, condition, landmarks in dl:
  landmarks *= 128
  for i in range(ims.shape[0]):
    im = ims[i]
    c = condition[i]
    l = landmarks[i]
    l = [keypoint1d_to_2d(l).cpu().numpy()]
    im = data_utils.denormalize_img(im)
    im = torch_utils.image_to_numpy(im, to_uint8=True)
    im = vis_utils.draw_faces_with_keypoints(im, None,l, False)
    c = data_utils.denormalize_img(c)
    c = torch_utils.image_to_numpy(c, to_uint8=True)
    to_save_ims.append(im)
    to_save_condition.append(c)
    if len(to_save_ims) == num_ims: break
  if len(to_save_ims) == num_ims: break
im = vis_utils.np_make_image_grid(to_save_ims + to_save_condition, 2)
plt.imsave(".debug/dataloader_test.jpg", im)
