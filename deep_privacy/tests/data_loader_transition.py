import matplotlib.pyplot as plt
import numpy as np
import torch
from deep_privacy.data_tools.dataloaders_v2 import load_dataset
from deep_privacy import torch_utils
from deep_privacy.data_tools import data_utils


start_imsize = 8
batch_size=32

dl_train, dl_val = load_dataset("fdf", batch_size, start_imsize, False, 14, True)

dl = dl_val
dl.update_next_transition_variable(1.0)
next(iter(dl))


for im, condition, landmark in dl:
    im = data_utils.denormalize_img(im)
    im = torch_utils.image_to_numpy(im, to_uint8=True)
    to_save1 = im
    break
to_save1 = np.concatenate(to_save1, axis=1)
dl_train, dl_val = load_dataset("fdf", batch_size, start_imsize*2, False, 14, True)
dl = dl_val
dl.update_next_transition_variable(0.0)
next(iter(dl))


for im, condition, landmark in dl:
    im = torch.nn.functional.avg_pool2d(im, 2)
    im = data_utils.denormalize_img(im)
    im = torch_utils.image_to_numpy(im, to_uint8=True)
    to_save2 = im
    break
to_save2 = np.concatenate(to_save2, axis=1)
print(to_save1.shape, to_save2.shape)
print("Diff:", abs(to_save1 - to_save2).sum())
to_save = np.concatenate((to_save1, to_save2), axis=0)
plt.imsave(".debug/transition_test_loader_downsample.jpg", to_save)




dl_train, dl_val = load_dataset("fdf", batch_size, start_imsize, False, 14, True)

dl = dl_val
dl.update_next_transition_variable(1.0)
next(iter(dl))


for im, condition, landmark in dl:
    im = torch.nn.functional.interpolate(im, scale_factor=2)
    im = data_utils.denormalize_img(im)
    im = torch_utils.image_to_numpy(im, to_uint8=True)
    to_save1 = im
    break
to_save1 = np.concatenate(to_save1, axis=1)
dl_train, dl_val = load_dataset("fdf", batch_size, start_imsize*2, False, 14, True)
dl = dl_val
dl.update_next_transition_variable(0.0)
next(iter(dl))


for im, condition, landmark in dl:
    im = data_utils.denormalize_img(im)
    im = torch_utils.image_to_numpy(im, to_uint8=True)
    to_save2 = im
    break
to_save2 = np.concatenate(to_save2, axis=1)
print(to_save1.shape, to_save2.shape)
print("Diff upsample:", abs(to_save1 - to_save2).sum())
to_save = np.concatenate((to_save1, to_save2), axis=0)
plt.imsave(".debug/transition_test_loader_upsample.jpg", to_save)