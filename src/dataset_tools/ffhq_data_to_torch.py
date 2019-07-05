import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import multiprocessing
import torch
from torchvision.transforms.functional import to_tensor
import math

SOURCE_DIR = "/lhome/haakohu/ffhq-dataset/data_unaligned"
ORIGINAL_IMAGE_SIZE = 128
SOURCE_IMG_DIR = os.path.join(SOURCE_DIR, "{}x{}".format(
    ORIGINAL_IMAGE_SIZE, ORIGINAL_IMAGE_SIZE))

# Loading CSV files
BBOX_FILE = os.path.join(SOURCE_DIR, "bbox_file.csv")
BBOX_FILE = pd.read_csv(BBOX_FILE)
BBOX_FILE = BBOX_FILE.set_index("image_id")
LANDMARKS_FILE = os.path.join(SOURCE_DIR, "landmarks_file.csv")
LANDMARKS_FILE = pd.read_csv(LANDMARKS_FILE)
LANDMARKS_FILE = LANDMARKS_FILE.set_index("image_id")

#LANDMARKS_TOTAL_FILE = os.path.join(SOURCE_DIR, "landmarks_total_file.csv")
#LANDMARKS_TOTAL_FILE = pd.read_csv(LANDMARKS_TOTAL_FILE)
#LANDMARKS_TOTAL_FILE = LANDMARKS_TOTAL_FILE.set_index("image_id")

TARGET_TORCH_DIR = os.path.join("data", "ffhq_unaligned_torch")
TARGET_IMAGE_DIR = os.path.join(TARGET_TORCH_DIR, "original")


def get_impath(image_id):
    imname = "{:05d}.png".format(image_id)
    impath = os.path.join(SOURCE_IMG_DIR, imname)
    return impath


def save_image_batch(idx, image_ids):
    imsizes = [4, 8, 16, 32, 64, 128]
    impaths = [os.path.join(SOURCE_IMG_DIR, get_impath(image_id))
               for image_id in image_ids]
    images = []
    for impath in impaths:
        images.append(plt.imread(impath))

    for imsize in imsizes:
        to_save = torch.zeros((len(impaths), 3, imsize, imsize),
                              dtype=torch.float32)
        for i, im in enumerate(images):
            im = im[:, :, :3]
            im = cv2.resize(im, (imsize, imsize),
                            interpolation=cv2.INTER_AREA)
            im = to_tensor(im)
            assert im.max() <= 1.0
            assert len(im.shape) == 3
            assert im.dtype == torch.float32
            to_save[i] = im
        target_dir = os.path.join(TARGET_IMAGE_DIR, str(imsize))
        target_path = os.path.join(target_dir, "{}.torch".format(str(idx)))
        os.makedirs(target_dir, exist_ok=True)
        torch.save(to_save, target_path)
        del to_save


def save_bounding_boxes(image_ids):
    target_dir = os.path.join(TARGET_TORCH_DIR, "bounding_box")
    os.makedirs(target_dir, exist_ok=True)
    # Get array to save
    bounding_boxes = BBOX_FILE.loc[image_ids].values
    bounding_boxes = bounding_boxes / ORIGINAL_IMAGE_SIZE

    imsizes = [4, 8, 16, 32, 64, 128]
    for imsize in tqdm(imsizes, desc="Saving bounding boxes"):
        bounding_boxes_resized = torch.from_numpy(bounding_boxes) * imsize
        target_path = os.path.join(target_dir, "{}.torch".format(imsize))
        torch.save(bounding_boxes_resized, target_path)
        del bounding_boxes_resized


def save_landmarks(image_ids):
    target_dir = os.path.join(TARGET_TORCH_DIR, "landmarks")
    landmarks = LANDMARKS_FILE.loc[image_ids].values
    landmarks = torch.from_numpy(landmarks) / ORIGINAL_IMAGE_SIZE
    os.makedirs(target_dir, exist_ok=True)
    imsizes = [4, 8, 16, 32, 64, 128]
    for imsize in tqdm(imsizes, desc="Saving landmarks"):

        landmarks_resized = landmarks * imsize

        target_path = os.path.join(target_dir, "{}.torch".format(imsize))
        torch.save(landmarks_resized, target_path)
        del landmarks_resized


def save_landmarks_total(image_ids):
    target_dir = os.path.join(TARGET_TORCH_DIR, "landmarks_total")
    landmarks = LANDMARKS_TOTAL_FILE.loc[image_ids].values
    landmarks = torch.from_numpy(landmarks) / ORIGINAL_IMAGE_SIZE
    os.makedirs(target_dir, exist_ok=True)
    imsizes = [4, 8, 16, 32, 64, 128]
    for imsize in tqdm(imsizes, desc="Saving landmarks"):
        landmarks_resized = landmarks * imsize
        target_path = os.path.join(target_dir, "{}.torch".format(imsize))
        torch.save(landmarks_resized, target_path)
        del landmarks_resized


def get_image_ids():
    landmark_ids = set(LANDMARKS_FILE.index.values)
    bbox_ids = set(BBOX_FILE.index.values)
    image_ids = list(landmark_ids.intersection(bbox_ids))
    image_ids.sort(key=lambda x: int(x))
    return image_ids


def dataset_to_torch():
    # Save real
    image_ids = get_image_ids()
    num_jobs = 200
    batch_size = math.ceil(len(image_ids) / num_jobs)
    jobs = []
    with multiprocessing.Pool(16) as p:
        for i in range(num_jobs):
            batch_ids = image_ids[i*batch_size: (i+1)*batch_size]
            j = p.apply_async(save_image_batch, (i, batch_ids))
            jobs.append(j)
        for j in tqdm(jobs, desc="Saving to torch"):
            j.get()

    save_bounding_boxes(image_ids)
    save_landmarks(image_ids)
    #save_landmarks_total(image_ids)

    f = open(os.path.join(TARGET_TORCH_DIR, "idx_to_imname.csv"), "w")
    f.write("image_id\n")
    f.write("\n".join([str(x) for x in image_ids]))
    f.close()


if __name__ == "__main__":
    dataset_to_torch()
