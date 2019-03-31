from torch import nn
import torch
import numpy as np
from apex import amp

# https://github.com/nvnbny/progressive_growing_of_gans/blob/master/modelUtils.py
class WSConv2d(nn.Module):
    """
    This is the wt scaling conv layer layer. Initialize with N(0, scale). Then 
    it will multiply the scale for every forward pass
    """
    def __init__(self, inCh, outCh, kernelSize, padding, gain=np.sqrt(2)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=inCh, out_channels=outCh, kernel_size=kernelSize, stride=1, padding=padding)

        # new bias to use after wscale
        self.bias = self.conv.bias
        self.bias = nn.Parameter(self.bias.view(1, self.bias.shape[0], 1, 1))
        self.conv.bias = None
        
        # calc wt scale
        convShape = list(self.conv.weight.shape)
        fanIn = np.prod(convShape[1:]) # Leave out # of op filters
        self.wtScale = gain/np.sqrt(fanIn)
        
        # init
        nn.init.normal_(self.conv.weight)
        nn.init.constant_(self.bias, val=0)
        self.name = '(inp = %s)' % (self.conv.__class__.__name__ + str(convShape))
        
    def forward(self, x):
        #return self.conv(x)
        return self.conv(x * self.wtScale) + self.bias

    def __repr__(self):
        return self.__class__.__name__ + self.name


class WSLinear(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bias = self.linear.bias
        self.linear.bias = None
        fanIn = in_dim
        self.wtScale = np.sqrt(2) / np.sqrt(fanIn)

        nn.init.normal_(self.linear.weight)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x):
        return self.linear(x * self.wtScale) + self.bias

class PixelwiseNormalization(nn.Module):

    def __init__(self):
        super().__init__()

    @amp.float_function
    def forward(self, x):
        factor = ((x**2 ).mean(dim=1, keepdim=True) + 1e-8)**0.5
        return x / factor

class UpSamplingBlock(nn.Module):
    def __init__(self):
        super(UpSamplingBlock, self).__init__()
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2)
