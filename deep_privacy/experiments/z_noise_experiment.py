import torch
import numpy as np
from deep_privacy import torch_utils
import os
from deep_privacy.inference import infer
from deep_privacy.data_tools.dataloaders import load_dataset_files, cut_bounding_box
import matplotlib.pyplot as plt


if __name__ == "__main__":
    generator, _, _, _, _ = infer.read_args()
    imsize = generator.current_imsize
    images, bounding_boxes, landmarks = load_dataset_files("data/fdf_png", imsize,
                                                           load_fraction=True)
    savedir = os.path.join(".debug", "test_examples", "z_noise_test")
    os.makedirs(savedir, exist_ok=True)
    generator.eval()
    num_iterations = 20
    zs = [generator.generate_latent_variable(1, "cuda", torch.float32)
          for i in range(num_iterations)]
    all_ims = []
    for idx in range(-5, -1):
        orig = images[idx]
        orig = np.array(orig)
        assert orig.dtype == np.uint8
        pose = landmarks[idx:idx+1]
        bbox = bounding_boxes[idx]

        to_save = orig.copy()

        truncation_levels = np.linspace(0.00, 3, num_iterations)
        final_images = []

        for i in range(num_iterations):
            im = orig.copy()
            im = cut_bounding_box(im, bbox, generator.transition_value)

            im = torch_utils.image_to_torch(im, cuda=True, normalize_img=True)
            im = generator(im, pose, zs[i].clone())

            im = torch_utils.image_to_numpy(im.squeeze(), to_uint8=True,
                                            denormalize=True)
            to_save = np.concatenate((to_save, im), axis=1)
            final_images.append(im)
        all_ims.append(to_save)
    all_ims = np.concatenate(all_ims, axis=0)

    imname = "result_image"
    savepath = os.path.join(savedir, f"{imname}.jpg")

    plt.imsave(savepath, all_ims)

    print("Results saved to:", savedir)