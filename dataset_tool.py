import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.transforms.functional import to_tensor
import torch
import tqdm


def save_dataset_to_torch_tensor(files_dirpath, target_path):
    print("Dirpath:", files_dirpath, "Target path:", target_path)
    files = glob.glob(os.path.join(files_dirpath, "*.jpg"))
    im = plt.imread(files[0])
    to_save = torch.zeros((len(files), 3, im.shape[1], im.shape[0]), dtype=torch.float32)
    for i, filepath in enumerate(tqdm.tqdm(files)):
        im = plt.imread(filepath)
        assert (im[:, :, 3] != 255).sum() == 0
        im = im[:, :, :3]
        im = to_tensor(im)
        assert im.max() <= 1.0
        assert im.dtype == torch.float32
        to_save[i] = im
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)
    torch.save(to_save, target_path)



if __name__ == "__main__":
    import os
    import glob
    folders = glob.glob("data/celebahq_gen/*")
    folders = [f for f in folders if os.path.isdir(f)]
    target_dir = os.path.join("data", "celebahq_torch")
    print(folders)
    for folder in folders:
        num = os.path.basename(folder)
        imsize = 2**(int(num))
        path = os.path.join(target_dir, "{}.torch".format(imsize))
        save_dataset_to_torch_tensor(folder, path)



    
