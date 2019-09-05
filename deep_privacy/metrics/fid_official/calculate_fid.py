import tqdm
import os
import shutil
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import torch
from deep_privacy import torch_utils, config_parser
from deep_privacy.data_tools.dataloaders import load_dataset
from deep_privacy.data_tools.data_utils import denormalize_img
from deep_privacy.utils import load_checkpoint
from deep_privacy.inference.infer import init_generator 


def read_args():
    config = config_parser.initialize_and_validate_config([
        {"name": "target_path", "default": ""}
    ])
    save_path = config.target_path
    if save_path == "":
        default_path = os.path.join(
            os.path.dirname(config.config_path),
            "fid_images"
        )
        print("Setting target path to default:", default_path)
        save_path = default_path
    model_name = config.config_path.split("/")[-2]
    ckpt = load_checkpoint(os.path.join("validation_checkpoints", model_name))
    #ckpt = load_checkpoint(os.path.join(
    #                                    os.path.dirname(config.config_path),
    #                                    "checkpoints"))
    generator = init_generator(config, ckpt)
    imsize = ckpt["current_imsize"]
    pose_size = config.models.pose_size
    return generator, imsize, save_path, pose_size


generator, imsize, save_path, pose_size = read_args()

batch_size = 128
dataloader_train, dataloader_val = load_dataset("fdf", batch_size, 128, True, pose_size, True )
dataloader_val.update_next_transition_variable(1.0)
fake_images = np.zeros((len(dataloader_val)*batch_size, imsize, imsize, 3),
                       dtype=np.uint8)
real_images = np.zeros((len(dataloader_val)*batch_size, imsize, imsize, 3),
                       dtype=np.uint8)
z = generator.generate_latent_variable(batch_size, "cuda", torch.float32).zero_()
with torch.no_grad():
    for idx, (real_data, condition, landmarks) in enumerate(tqdm.tqdm(dataloader_val)):

        fake_data = generator(condition, landmarks, z.clone())
        fake_data = torch_utils.image_to_numpy(fake_data, to_uint8=True, denormalize=True)
        real_data = torch_utils.image_to_numpy(real_data, to_uint8=True, denormalize=True)
        start_idx = idx * batch_size
        end_idx = (idx+1) * batch_size

        real_images[start_idx:end_idx] = real_data
        fake_images[start_idx:end_idx] = fake_data

generator.cpu()
del generator

if os.path.isdir(save_path):
    shutil.rmtree(save_path)

os.makedirs(os.path.join(save_path, "real"))
os.makedirs(os.path.join(save_path, "fake"))

def save_im(fp, im):
    plt.imsave(fp, im)

def save_images(images, path):
    
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        for idx, im in enumerate(tqdm.tqdm(images, desc="Starting jobs")):
            fp = os.path.join(path, "{}.jpg".format(idx))
            jobs.append(pool.apply_async(save_im, (fp, im)))
        for j in tqdm.tqdm(jobs, desc="Saving images"):
            j.get()
save_images(real_images, os.path.join(save_path, "real"))
save_images(fake_images, os.path.join(save_path, "fake"))

