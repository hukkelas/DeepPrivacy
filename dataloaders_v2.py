import torch
import os
import utils
import glob
import numpy as np
from torchvision import transforms


class DeepPrivacyDataset(torch.utils.data.Dataset):

    def __init__(self, images, bounding_boxes, landmarks):
        self.images = images
        self.imsize = self.images.shape[1]
        self.bounding_boxes = bounding_boxes
        self.landmarks = landmarks
        assert self.landmarks.shape[0] == self.bounding_boxes.shape[0]
        assert self.bounding_boxes.shape[0] == self.images.shape[0], "The number \
            of samples of images doesn't match number of bounding boxes. Images: {}, bbox: {}".format(self.images.shape[0], self.bounding_boxes.shape[0])
        expected_imshape = (self.imsize, self.imsize, 3)
        assert self.images.shape[1:] == expected_imshape, "Shape was: {}. Expected: {}".format(
            self.images.shape[1:], expected_imshape)
        print("Dataset loaded. Number of samples:", self.images.shape)

    def __getitem__(self, index):
        im = self.images[index]
        im = transforms.functional.to_pil_image(im)
        bbox = self.bounding_boxes[index]
        bbox = bounding_box_data_augmentation(bbox, self.imsize, 0.2)
        landmarks = self.landmarks[index]
        if np.random.rand() > 0.5:
            im = transforms.functional.hflip(im)
            x = landmarks[range(landmarks.shape[0], 2)]
            landmarks[range(landmarks.shape[0], 2)] = 1 - x
            bbox[[0, 2]] = self.imsize - bbox[[2, 0]]
        im = np.asarray(im)
        condition = im.copy()
        condition = cut_bounding_box(condition, bbox)

        return im, condition, landmarks

    def __len__(self):
        return self.images.shape[0]


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


def load_numpy_files(dirpath):
    images = []
    files = glob.glob(os.path.join(dirpath, "*.npy"))
    files.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
    assert len(files) > 0, "Empty directory: " + dirpath
    for fpath in files:
        assert os.path.isfile(fpath), "Is not file: " + fpath
        ims = np.load(fpath)
        assert ims.dtype == np.uint8, "Was: {}".format(ims.dtype)
        images.append(ims)
    images = np.concatenate(images, axis=0)
    return images


def load_ffhq_condition(batch_size, imsize=128): 
    return ConditionedCelebADataset("data/ffhq_torch", imsize, batch_size, landmarks_total=False)


def load_dataset(dirpath, imsize, batch_size):
    images = load_numpy_files(os.path.join(dirpath, "original", str(imsize)))
    bounding_box_filepath = os.path.join(
        dirpath, "bounding_box", "{}.torch".format(imsize))
    assert os.path.isfile(bounding_box_filepath), "Did not find the bounding box data. Looked in: {}".format(
        bounding_box_filepath)
    bounding_boxes = torch.load(bounding_box_filepath).long()

    landmark_filepath = os.path.join(
        dirpath, "landmarks", "{}.torch".format(imsize))
    assert os.path.isfile(
        landmark_filepath), "Did not find the landmark data. Looked in: {}".format(landmark_filepath)
    landmarks = torch.load(landmark_filepath).float() / imsize
    validation_size = int(0.05*len(images))
    images_train, images_val = images[:-validation_size], images[-validation_size:]
    bbox_train, bbox_val = bounding_boxes[:-validation_size], bounding_boxes[-validation_size:]
    lm_train, lm_val = landmarks[:-validation_size], landmarks[-validation_size:]

    dataset_train = DeepPrivacyDataset(images_train, bbox_train, lm_train)
    dataset_val = DeepPrivacyDataset(images_val, bbox_val, lm_val)
    print("LEN DATASET VAL:", len(dataset_val), len(images_val))
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=8,
                                                   drop_last=True,
                                                   pin_memory=True,
                                                   collate_fn=fast_collate)
    dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=2,
                                                 drop_last=True,
                                                 pin_memory=True,
                                                 collate_fn=fast_collate)
    return dataloader_train, dataloader_val


def load_celeba_condition(batch_size, imsize=128):
    dirpath = os.path.join("data", "celeba_numpy")
    return load_dataset(dirpath, imsize, batch_size)


def bounding_box_data_augmentation(bounding_boxes, imsize, percentage):
    # Data augment width and height by percentage of width.
    # Bounding box will preserve its center.
    shrink_percentage = np.random.uniform()
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


def cut_bounding_box(condition, bounding_boxes):
    x0 = bounding_boxes[0]
    y0 = bounding_boxes[1]
    x1 = bounding_boxes[2]
    y1 = bounding_boxes[3]
    previous_image = condition[y0:y1, x0:x1]
    mean = previous_image.mean()
    std = previous_image.std()
    replacement = np.random.normal(mean, std,
                                   size=previous_image.shape)*255
    previous_image[:, :, :] = replacement.astype(np.uint8)
    return condition
