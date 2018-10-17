import torch
import torch.nn as nn
from torch.autograd import Variable


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        def conv_bn_relu(in_dim, out_dim, kernel_size, padding=0):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, [kernel_size, kernel_size], padding=padding),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU()
            )

        self.model = nn.Sequential(
            conv_bn_relu(100,128, 4, 3),
            conv_bn_relu(128, 128, 3, 1),

            # Upsample: 8x8
            nn.Upsample(scale_factor=2),
            conv_bn_relu(128, 64, 3, 1),
            conv_bn_relu(64, 64, 3, 1),

            # Upsample 16x16
            nn.Upsample(scale_factor=2),
            conv_bn_relu(64,32,3,1),
            conv_bn_relu(32, 32, 3, 1),

            # Upsample 32x32
            nn.Upsample(scale_factor=2),
            conv_bn_relu(32, 16, 3),
            conv_bn_relu(16, 16, 3),
            nn.Conv2d(16, 1, 1),
            nn.Tanh()
        )

    # x: Bx1x1x512
    def forward(self, x):
        x = x.view((x.shape[0], -1, 1, 1))
        x = self.model(x)
        return x


class UpSamplingBlock(nn.Module):
    def __init__(self):
        super(UpSamplingBlock, self).__init__()
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        def conv_module(dim_in, dim_out, kernel_size, padding):
            return nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size, padding=padding),
                nn.LeakyReLU()
            )
        self.model = nn.Sequential(
            conv_module(1, 16, 1, 2),
            conv_module(16, 16, 3, 1),
            conv_module(16, 32, 3, 1),

            # Downsample 16x16
            nn.AvgPool2d([2, 2]), 
            conv_module(32, 32, 3, 1),
            conv_module(32, 64, 3, 1),

            # Downsample 8x8
            nn.AvgPool2d([2, 2]),
            conv_module(64, 64, 3, 1),
            conv_module(64, 128, 3, 1),

            # Downsample 4x4
            nn.AvgPool2d([2, 2]),
            conv_module(128, 128, 3, 1),
            conv_module(128, 128, 4, 0) 
        )
        self.linear = nn.Sequential(
            nn.Linear(128, 1),
            #nn.Sigmoid()
        )
        

    # x: Bx1x1x512
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 128)
        x = self.linear(x)
        return x



if __name__ == "__main__":
    G = Generator()
    x = torch.zeros((4, 128, 1, 1)).uniform_()
    x = G(x)
    print(x.shape)

    D = Discriminator()
    
    x = D(x)
    print(x.shape)
    print(x)


    