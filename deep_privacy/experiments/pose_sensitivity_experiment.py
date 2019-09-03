import torch
import numpy as np
from deep_privacy import torch_utils
from deep_privacy.visualization import utils as vis_utils
import os
from deep_privacy.inference import infer
from deep_privacy.data_tools.dataloaders_v2 import load_dataset_files, cut_bounding_box
import matplotlib.pyplot as plt


if __name__ == "__main__":
    generator, _, _, _, _ = infer.read_args()
    imsize = generator.current_imsize
    images, bounding_boxes, landmarks = load_dataset_files("data/fdf", imsize,
                                                           load_fraction=True)

    batch_size = 128

    savedir = os.path.join(".debug","test_examples", "pose_sensitivity_experiment")
    os.makedirs(savedir, exist_ok=True)
    num_iterations = 20
    ims_to_save = []
    percentages = [0] + list(np.linspace(-0.3, 0.3, num_iterations-1))
    z = generator.generate_latent_variable(1, "cuda", torch.float32).zero_()
    for idx in range(-5, -1):
        orig = images[idx]
        orig_pose = landmarks[idx:idx+1]
        bbox = bounding_boxes[idx].clone()
        assert orig.dtype == np.uint8

        to_save = orig.copy()
        to_save = np.tile(to_save, (2, 1, 1))

        truncation_levels = np.linspace(0.00, 3, num_iterations)

        for i in range(num_iterations):
            im = orig.copy()
            orig_to_save = im.copy()

            p = percentages[i]
            pose = orig_pose.clone()

            rand = torch.rand_like(pose) - 0.5
            rand = (rand / 0.5) * p
            pose += rand

            im = cut_bounding_box(im, bbox, generator.transition_value)

            keypoints = (pose*imsize).long()
            keypoints = infer.keypoint_to_numpy(keypoints)
            orig_to_save = vis_utils.draw_faces_with_keypoints(
                orig_to_save, None, [keypoints],
                radius=3)

            im = torch_utils.image_to_torch(im, cuda=True, normalize_img=True)
            im = generator(im, pose, z.clone())
            im = torch_utils.image_to_numpy(im.squeeze(), to_uint8=True, denormalize=True)

            im = np.concatenate((orig_to_save.astype(np.uint8), im), axis=0)
            to_save = np.concatenate((to_save, im), axis=1)
        ims_to_save.append(to_save)
    savepath = os.path.join(savedir, f"result_image.jpg")
    
    ims_to_save = np.concatenate(ims_to_save, axis=0)
    plt.imsave(savepath, ims_to_save)

    print("Results saved to:", savedir)