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