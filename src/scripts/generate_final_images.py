import torch 
from utils import load_checkpoint
from scripts.utils import image_to_numpy, init_generator, get_model_name
from train import denormalize_img, preprocess_images
from dataloaders import  load_celeba_condition
import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    model_name = get_model_name()
    ckpt_path = os.path.join("checkpoints", model_name)
    ckpt = load_checkpoint(ckpt_path)
    g = init_generator(ckpt)
    g.eval()
    
    
    dataloader = load_celeba_condition(10, ckpt["current_imsize"])
    
    images = []
    conditions = []
    originals = []
    transition_value = ckpt["transition_variable"]
    for orig, z in dataloader.validation_set_generator():
        z = preprocess_images(z, transition_value)

        d = g(z, None)
        d = denormalize_img(d)
        c = denormalize_img(z)
        conditions.append(c.data.cpu())
        images.append(d.data.cpu())
        originals.append(orig.data.cpu())
        
    images = torch.cat(images, dim=0)
    originals = torch.cat(originals, dim=0)
    conditions = torch.cat(conditions, dim=0)
    images = image_to_numpy(images)
    originals = image_to_numpy(originals)
    conditions = image_to_numpy(conditions)

    num_rows = 20
    num_cols = 5
    imsize = 128
    to_save = np.zeros((num_rows*imsize, num_cols*imsize*3 + 3*(num_cols - 1), 3))

    for row in range(num_rows):
        for col in range(num_cols):
            idx = row*num_cols + col
            row_start = row*imsize
            row_end = (row+1)*imsize
            col_start = col*imsize*3 + 2*col
            col_end = col_start + 3*imsize

            orig = images[idx]
            cond = conditions[idx]
            im = np.concatenate((originals[idx], cond, orig), axis=1)
            to_save[row_start:row_end, col_start:col_end] = im
    example_dir = os.path.join("test_examples", model_name)
    os.makedirs(example_dir, exist_ok=True)
    filepath = os.path.join(example_dir, "random_generated_images.jpg")
    plt.imsave(filepath, to_save)
    print("Saved resulting image to:", filepath)

