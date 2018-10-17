import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch


def load_mnist(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])])
    imagenet_data = datasets.MNIST('data/mnist_data', 
                                train=True, 
                                download=True,
                                transform=transform)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4)
    return data_loader