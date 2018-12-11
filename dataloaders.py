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
    
class ConditionedCelebADataset:

    def __init__(self, dirpath, imsize, batch_size):
        self.images = load_torch_files(os.path.join(dirpath, "original", str(imsize)))
        self.conditional_images = load_torch_files(os.path.join(dirpath, "anonymized", str(imsize)))
        assert self.images.shape == self.conditional_images.shape
        expected_shape = (3, imsize, imsize)
        assert self.images.shape[1:] == expected_shape, "Shape was: {}. Expected: {}".format(self.images.shape[1:], expected_shape)
        print("Dataset loaded. Number of samples:", self.images.shape)
        self.n_samples = self.images.shape[0]
        self.batch_size = batch_size
        self.indices = torch.LongTensor(self.batch_size)
        self.ones = torch.ones(self.batch_size, dtype=torch.long)
        self.max = self.n_samples // self.batch_size
        self.validation_size = int(self.n_samples*0.05)

    
    def __len__(self):
        return self.images.shape[0] // self.batch_size

    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self):
        if self.n > self.max:
            raise StopIteration
        self.n += 1
        indices = self.indices.random_(0, self.n_samples - self.validation_size)
        images = self.images[indices]
        condition = self.conditional_images[indices]
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
            yield self.images[start_idx:end_idx], self.conditional_images[start_idx:end_idx]