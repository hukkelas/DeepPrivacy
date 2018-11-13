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
    celeba_data = CelebADataset("data/celebahq_torch", imsize)
    data_loader = torch.utils.data.DataLoader(celeba_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=8,
                                            pin_memory=True)
    return data_loader


class CelebADataset(torch.utils.data.Dataset):


    def __init__(self, dirpath, imsize):
        filepath = os.path.join(dirpath, "{}.torch".format(imsize))
        assert os.path.isfile(filepath), "Did not find file in filepath:{}".format(filepath)
        images = torch.load(filepath)
        assert images.dtype == torch.float32
        expected_shape = (3, imsize, imsize)
        assert images.shape[1:] == expected_shape, "Shape was: {}. Expected: {}".format(images.shape[1:], expected_shape)
        self.images = images
    
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        image = self.images[idx]
        if np.random.random() > 0.5: # flip
            image = utils.flip_horizontal(image)
        return image, torch.tensor(1)

    


