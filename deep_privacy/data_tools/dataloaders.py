import torch
import os
import deep_privacy.models.utils as model_utils
import glob
import numpy as np
import tqdm
import multiprocessing
from torchvision import transforms
from deep_privacy.data_tools.data_utils import DataPrefetcher
from PIL import Image

MAX_VALIDATION_SIZE = 50000


def load_dataset(dataset, batch_size, imsize, full_validation, pose_size, load_fraction=False):
    if dataset == "celeba":
        dirpath = os.path.join("data", "celeba_numpy")
        return _load_dataset(dirpath, imsize, batch_size, full_validation, load_fraction, pose_size)
    if dataset == "ffhq":
        raise NotImplementedError
    if dataset == "yfcc100m":
        dirpath = os.path.join("data", "yfcc100m_torch")
        return _load_dataset(dirpath, imsize, batch_size, full_validation, load_fraction, pose_size)
    if dataset == "yfcc100m128":
        dirpath = os.path.join("data", "yfcc100m_torch_fix_transition")
        return _load_dataset(dirpath, imsize, batch_size, full_validation, load_fraction, pose_size)
    if dataset == "fdf":
        dirpath = os.path.join("data", "fdf_png")
        return _load_dataset(dirpath, imsize, batch_size, full_validation, load_fraction, pose_size)
    raise AssertionError("Dataset was incorrect", dataset)


class DeepPrivacyDataset(torch.utils.data.Dataset):

    def __init__(self, images, bounding_boxes, landmarks, augment_data):
        self.augment_data = augment_data
        self.images = images
        self.imsize = self.images[0].size[0]
        self.bounding_boxes = bounding_boxes
        self.landmarks = landmarks
        assert self.landmarks.shape[0] == self.bounding_boxes.shape[0]
        assert self.bounding_boxes.shape[0] == len(self.images), "The number \
            of samples of images doesn't match number of bounding boxes. Images: {}, bbox: {}".format(len(self.images), self.bounding_boxes.shape[0])
        print("Dataset loaded. Number of samples:", len(self.images))

    def __getitem__(self, index):
        im = self.images[index]
        landmarks = self.landmarks[index].clone()
        bbox = self.bounding_boxes[index].clone()
        if self.augment_data:
            bbox = bounding_box_data_augmentation(bbox, self.imsize, 2/100)
            if np.random.rand() > 0.5:
                im = transforms.functional.hflip(im)
                x = landmarks[range(0, landmarks.shape[0], 2)]
                landmarks[range(0, landmarks.shape[0], 2)] = 1 - x
                bbox[[0, 2]] = self.imsize - bbox[[2, 0]]
        im = np.array(im)
        condition = im.copy()
        condition = cut_bounding_box(condition, bbox, self.transition_variable)
        return im, condition, landmarks

    def __len__(self):
        return len(self.images)


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    conds = [img[1] for img in batch]
    landmarks = torch.stack([lm[2] for lm in batch])
    w = imgs[0].shape[0]
    h = imgs[0].shape[1]
    images = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    conditions = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, (img, cond) in enumerate(zip(imgs, conds)):
        nump_array = np.rollaxis(img, 2)
        conditions[i] += torch.from_numpy(np.rollaxis(cond, 2))
        images[i] += torch.from_numpy(nump_array)
    return images, conditions, landmarks


def read_image(impath):
    im = Image.open(impath)
    im.load()
    return im


def load_images(dirpath, load_fraction):
    images = []
    files = glob.glob(os.path.join(dirpath, "*.png"))
    files.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
    if load_fraction:
        files = files[:1000] + files[-MAX_VALIDATION_SIZE:]
    assert len(files) > 0, "Empty directory: " + dirpath
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        for fpath in files:
            assert os.path.isfile(fpath), "Is not file: " + fpath
            jobs.append(
                pool.apply_async(read_image, (fpath, )))
        for job in tqdm.tqdm(jobs, desc="Reading images"):
            images.append(job.get())

    return images


def load_dataset_files(dirpath, imsize, load_fraction):
    print("loading images from:", dirpath)
    images = load_images(os.path.join(dirpath, "images", str(imsize)), load_fraction)
    bounding_box_filepath = os.path.join(
        dirpath, "bounding_box", "{}.npy".format(imsize))
    assert os.path.isfile(bounding_box_filepath), "Did not find the bounding box data. Looked in: {}".format(
        bounding_box_filepath)
    bounding_boxes = np.load(bounding_box_filepath)
    bounding_boxes = torch.from_numpy(bounding_boxes).long() 

    landmark_filepath = os.path.join(
        dirpath, "landmarks", "{}.npy".format(imsize))
    assert os.path.isfile(
        landmark_filepath), "Did not find the landmark data. Looked in: {}".format(landmark_filepath)
    landmarks = np.load(landmark_filepath)
    landmarks = torch.from_numpy(landmarks).float() / imsize
    if load_fraction:
        bounding_boxes = torch.cat([bounding_boxes[:len(images) - MAX_VALIDATION_SIZE],
                                    bounding_boxes[-MAX_VALIDATION_SIZE:]])
        landmarks = torch.cat([landmarks[:len(images) - MAX_VALIDATION_SIZE],
                               landmarks[-MAX_VALIDATION_SIZE:]]) 
    return images, bounding_boxes, landmarks


def _load_dataset(dirpath, imsize, batch_size, full_validation, load_fraction, pose_size):
    images, bounding_boxes, landmarks = load_dataset_files(dirpath, imsize,
                                                           load_fraction)
    if full_validation:
        validation_size = MAX_VALIDATION_SIZE
    else:
        validation_size = 10000

    # Keep out 50,000 images for final validation.
    images_train, images_val = images[:-MAX_VALIDATION_SIZE], images[-validation_size:]
    bbox_train, bbox_val = bounding_boxes[:-MAX_VALIDATION_SIZE], bounding_boxes[-validation_size:]
    lm_train, lm_val = landmarks[:-MAX_VALIDATION_SIZE], landmarks[-validation_size:]

    dataset_train = DeepPrivacyDataset(images_train, bbox_train, lm_train, True)
    dataset_val = DeepPrivacyDataset(images_val, bbox_val, lm_val, False)
    print("LEN DATASET VAL:", len(dataset_val), len(images_val))

    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=16,
                                                   drop_last=True,
                                                   pin_memory=True,
                                                   collate_fn=fast_collate)
    dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=8,
                                                 drop_last=True,
                                                 pin_memory=True,
                                                 collate_fn=fast_collate)
    dataloader_train = DataPrefetcher(dataloader_train, pose_size, dataset_train)
    dataloader_val = DataPrefetcher(dataloader_val, pose_size, dataset_val)
    return dataloader_train, dataloader_val


def bounding_box_data_augmentation(bounding_boxes, imsize, percentage):
    # Data augment width and height by percentage of width.
    # Bounding box will preserve its center.
    shrink_percentage = np.random.uniform(-percentage, percentage)
    width = (bounding_boxes[2] - bounding_boxes[0]).float()
    height = (bounding_boxes[3] - bounding_boxes[1]).float()

    # Can change 10% in each direction
    width_diff = shrink_percentage * width
    height_diff = shrink_percentage * height
    bounding_boxes[0] -= width_diff.long()
    bounding_boxes[1] -= height_diff.long()
    bounding_boxes[2] += width_diff.long()
    bounding_boxes[3] += height_diff.long()
    # Ensure that bounding box is within image
    bounding_boxes[0] = max(0, bounding_boxes[0])
    bounding_boxes[1] = max(0, bounding_boxes[1])
    bounding_boxes[2] = min(imsize, bounding_boxes[2])
    bounding_boxes[3] = min(imsize, bounding_boxes[3])
    return bounding_boxes


def cut_bounding_box(condition, bounding_boxes, transition_variable):
    assert 1 <= condition.max() <= 255
    bounding_boxes = bounding_boxes.clone()
    if transition_variable != 1:
        bounding_boxes_0 = bounding_boxes // 2 * 2
        bounding_boxes = model_utils.get_transition_value(bounding_boxes_0.float(), bounding_boxes.float(), transition_variable).long()

    x0, y0, x1, y1 = [k.item() for k in bounding_boxes]
    if x0 >= x1 or y0 >= y1:
        return condition

    condition[y0:y1, x0:x1, :] = 128
    return condition
