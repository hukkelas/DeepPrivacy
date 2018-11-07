import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch


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
    transform = [
        transforms.CenterCrop((128, 128))
    ]
    if imsize != 128:
        transform +=  [transforms.Resize([imsize, imsize])]
    transform += [
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ]
    transform = transforms.Compose(transform)
    imagenet_data = datasets.ImageFolder("data/img_align_celeba", transform=transform)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4)
    return data_loader