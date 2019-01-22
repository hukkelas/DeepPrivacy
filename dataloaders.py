import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import os
import numpy as np
import utils
import glob

def load_mnist(batch_size, imsize=32):
    transform = [
        transforms.Pad(2)
    ]
    if imsize != 32:
        transform +=  [transforms.Resize([imsize, imsize])]
    transform += [
        transforms.ToTensor(),
    ]
    transform = transforms.Compose(transform)
    imagenet_data = datasets.MNIST('data/mnist_data', 
                                train=True, 
                                download=True,
                                transform=transform)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4)
    return data_loader



def load_cifar10(batch_size, imsize=32):
    transform = []
    if imsize != 32:
        transform +=  [transforms.Resize([imsize, imsize])]
    transform += [
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
    transform = transforms.Compose(transform)
    imagenet_data = datasets.CIFAR10('data/cifar10', 
                                train=True, 
                                download=True,
                                transform=transform)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4)
    return data_loader

def load_pokemon(batch_size, imsize=96):
    transform = []
    if imsize != 96:
        transform +=  [transforms.Resize([imsize, imsize])]
    transform += [
        transforms.RandomHorizontalFlip(.5),
        transforms.ToTensor(),
    ]
    transform = transforms.Compose(transform)
    imagenet_data = datasets.ImageFolder("data/pokemons", transform=transform)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4)
    return data_loader


def load_celeba(batch_size, imsize=128):
    return CelebAGenerator("data/celebahq_torch", imsize, batch_size)
class CelebAGenerator:

    def __init__(self, dirpath, imsize, batch_size):
        filepath = os.path.join(dirpath, "{}.torch".format(imsize))
        assert os.path.isfile(filepath), "Did not find file in filepath:{}".format(filepath)
        images = torch.load(filepath)
        assert images.dtype == torch.float32
        expected_shape = (3, imsize, imsize)
        assert images.shape[1:] == expected_shape, "Shape was: {}. Expected: {}".format(images.shape[1:], expected_shape)
        self.images = images
        self.n_samples = images.shape[0]
        self.batch_size = batch_size
        self.indices = torch.LongTensor(self.batch_size)
        self.ones = torch.ones(self.batch_size, dtype=torch.long)
        self.max = self.n_samples // self.batch_size

    
    def __len__(self):
        return self.images.shape[0]

    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self):
        if self.n > self.max:
            raise StopIteration
        self.n += 1
        indices = self.indices.random_(0, self.n_samples)
        images = self.images[indices]
        to_flip = torch.rand(self.batch_size) > 0.5
        try:
            images[to_flip] = utils.flip_horizontal(images[to_flip])
        except:
            print("failed")
        return images, self.ones


    


def load_torch_files(dirpath):
    images = []
    files = glob.glob(os.path.join(dirpath, "*.torch"))
    files.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
    assert len(files) > 0, "Empty directory: " + dirpath
    for fpath in files:
        assert os.path.isfile(fpath), "Is not file: " + fpath
        ims = torch.load(fpath)
        assert ims.dtype == torch.float32, "Was: {}".format(ims.dtype)
        images.append(ims)
    images = torch.cat(images, dim=0)
    return images


def load_celeba_condition(batch_size, imsize=128):
    return ConditionedCelebADataset("data/celeba_torch", imsize, batch_size)



def bounding_box_data_augmentation(bounding_boxes, imsize, percentage):
    # Data augment width and height by percentage of width.
    # Bounding box will preserve its center.
    
    width_diff = torch.zeros((bounding_boxes.shape[0])).float()
    height_diff = torch.zeros_like(width_diff)
    width = (bounding_boxes[:, 2] - bounding_boxes[:, 0]).float()
    height = (bounding_boxes[:, 3] - bounding_boxes[:, 1]).float()

    # Can change 10% in each direction
    width_diff = width_diff.uniform_(-percentage, percentage) * width

    height_diff = height_diff.uniform_(-percentage, percentage) * height
    bounding_boxes[:, 0] -= width_diff.long()
    bounding_boxes[:, 1] -= height_diff.long()
    bounding_boxes[:, 2] += width_diff.long()
    bounding_boxes[:, 3] += height_diff.long()
    # Ensure that bounding box is within image
    bounding_boxes[:, 0][bounding_boxes[:, 0] < 0] = 0
    bounding_boxes[:, 1][bounding_boxes[:, 1] < 0] = 0
    bounding_boxes[:, 2][bounding_boxes[:, 2] > imsize] = imsize
    bounding_boxes[:, 3][bounding_boxes[:, 3] > imsize] = imsize
    return bounding_boxes

def bbox_to_coordinates(bounding_boxes):
    x0 = bounding_boxes[:, 0]
    y0 = bounding_boxes[:, 0]
    x1 = x0 + bounding_boxes[:, 2]
    y1 = y0 + bounding_boxes[:, 3]
    bounding_boxes[:, 2] = x1
    bounding_boxes[:, 3] = y1
    return bounding_boxes    


def cut_bounding_box(condition, bounding_boxes):
    x0 = bounding_boxes[:, 0]
    y0 = bounding_boxes[:, 0]
    x1 = bounding_boxes[:, 2]
    y1 = bounding_boxes[:, 3]
    for i in range(condition.shape[0]):
        condition[i, :, y0[i]:y1[i], x0[i]:x1[i]] = 0
    return condition


class ConditionedCelebADataset:

    def __init__(self, dirpath, imsize, batch_size):
        self.images = load_torch_files(os.path.join(dirpath, "original", str(imsize)))
        bounding_box_filepath = os.path.join(dirpath, "bounding_box", "{}.torch".format(imsize))
        assert os.path.isfile(bounding_box_filepath), "Did not find the bounding box data. Looked in: {}".format(bounding_box_filepath)
        self.bounding_boxes = torch.load(bounding_box_filepath)
        #self.bounding_boxes = bbox_to_coordinates(self.bounding_boxes)
        assert self.bounding_boxes.shape[0] == self.images.shape[0], "The number \
            of samples of images doesn't match number of bounding boxes. Images: {}, bbox: {}".format(self.images.shape[0], self.bounding_boxes.shape[0])
        expected_imshape = (3, imsize, imsize)
        assert self.images.shape[1:] == expected_imshape, "Shape was: {}. Expected: {}".format(self.images.shape[1:], expected_imshape)
        print("Dataset loaded. Number of samples:", self.images.shape)
        self.n_samples = self.images.shape[0]
        self.batch_size = batch_size
        self.indices = torch.LongTensor(self.batch_size)
        self.batches_per_epoch = self.n_samples // self.batch_size
        self.validation_size = int(self.n_samples*0.05)
        self.imsize = imsize

    
    def __len__(self):
        return self.images.shape[0] // self.batch_size

    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self):
        if self.n > self.batches_per_epoch:
            raise StopIteration
        self.n += 1
        indices = self.indices.random_(0, self.n_samples - self.validation_size)
        images = self.images[indices]
        bounding_boxes = self.bounding_boxes[indices]
        bounding_boxes = bounding_box_data_augmentation(bounding_boxes, self.imsize, 0.1)
        condition = images.clone()
        condition = cut_bounding_box(condition, bounding_boxes)
        to_flip = torch.rand(self.batch_size) > 0.5
        try:
            images[to_flip] = utils.flip_horizontal(images[to_flip])
            condition[to_flip] = utils.flip_horizontal(condition[to_flip])
        except:
            print("failed")
        return images, condition
    
    def validation_set_generator(self):
        validation_iters = self.validation_size // self.batch_size
        validation_offset = self.n_samples - self.validation_size
        for i in range(validation_iters):
            start_idx = validation_offset + i*self.batch_size
            end_idx = validation_offset + (i+1)*self.batch_size
            images = self.images[start_idx:end_idx]
            bounding_boxes = self.bounding_boxes[start_idx:end_idx]
            
            condition = images.clone()
            condition = cut_bounding_box(condition, bounding_boxes)
            yield images, condition
