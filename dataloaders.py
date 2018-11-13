import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import os
import numpy as np
import utils

def load_mnist(batch_size, imsize=32):
    transform = [
        transforms.Pad(2)
    ]
    if imsize != 32:
        transform +=  [transforms.Resize([imsize, imsize])]
    transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
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
        images[to_flip] = utils.flip_horizontal(images[to_flip])
        return images, self.ones


    

