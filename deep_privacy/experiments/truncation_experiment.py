import numpy as np
import matplotlib.pyplot as plt
import os
from deep_privacy import torch_utils
from deep_privacy.inference import infer
from deep_privacy.data_tools.dataloaders import load_dataset_files, cut_bounding_box


def truncated_z(z, x_in, generator, truncation_level):
    if truncation_level == 0:
        return z.zero_()
    while z.abs().max() >= truncation_level:
        mask = z.abs() >= truncation_level
        z[mask] = generator.generate_latent_variable(x_in)[mask]
    return z


if __name__ == "__main__":
    generator, _, _, _, _ = infer.read_args()
    imsize = generator.current_imsize
    use_truncation = True
    orig_z = None
    num_iterations = 20

    images, bounding_boxes, landmarks = load_dataset_files("data/fdf", imsize,
                                                           load_fraction=True)
    savedir = os.path.join(".debug","test_examples", "z_truncation")
    os.makedirs(savedir, exist_ok=True)
    generator.eval()
    all_ims = []
    for idx in range(-5, -1):
        orig = images[idx]
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
            if use_truncation:
                truncation_level = truncation_levels[i]
                if orig_z is None:
                    orig_z = generator.generate_latent_variable(im)
                z = truncated_z(orig_z.clone(), im, generator, truncation_level)
                im = generator(im, pose, z)
            else:
                im = generator(im, pose)

            im = torch_utils.image_to_numpy(im.squeeze(), to_uint8=True, denormalize=True)
            to_save = np.concatenate((to_save, im), axis=1)
            final_images.append(im)
        all_ims.append(to_save)
    all_ims = np.concatenate(all_ims, axis=0)

    imname = "result_image_trunc" if use_truncation else "result_image"
    savepath = os.path.join(savedir, f"{imname}.jpg")

    plt.imsave(savepath, all_ims)

    print("Results saved to:", savedir)