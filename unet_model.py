
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import to_cuda, init_weights
from torchsummary import summary
import numpy as np


class MinibatchStdLayer(nn.Module):

    def __init__(self):
        super(MinibatchStdLayer, self).__init__()

    def forward(self, x, group_size=4):
        group_size = min(group_size, x.shape[0]) # group_size must be smaller than minibatch size
        channels, height, width = x.shape[1:]
        y = x.view(group_size, -1, *x.shape[1:]) # Add extra "group" dimension and let minibatch size compensate
        y = y.float()
        y -= y.mean(dim=0, keepdim=True)
        y = y.pow(2).mean(dim=0)
        y = torch.sqrt(y + 1e-8)
        # Mean over minibatch, height and width
        y = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        y = y.to(x.dtype)
        # Tiling over (mb_size, channels, height, width)
        y = y.repeat(group_size, 1, height, width)
        return torch.cat((x, y), dim=1)


# https://github.com/nvnbny/progressive_growing_of_gans/blob/master/modelUtils.py
class WSConv2d(nn.Module):
    """
    This is the wt scaling conv layer layer. Initialize with N(0, scale). Then 
    it will multiply the scale for every forward pass
    """
    def __init__(self, inCh, outCh, kernelSize, stride, padding, gain=np.sqrt(2)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=inCh, out_channels=outCh, kernel_size=kernelSize, stride=stride, padding=padding)
        
        # new bias to use after wscale
        self.bias = self.conv.bias
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
        return self.conv(x) * self.wtScale + self.bias.view(1, self.bias.shape[0], 1, 1)

    def __repr__(self):
        return self.__class__.__name__ + self.name

        
class EqualizedConv2D(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size, padding):
        super(EqualizedConv2D, self).__init__()

        #self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, padding=padding)
        #self.conv.apply(torch.nn.init.he)
        #self.conv.apply(init_weights)
        #self.conv.bias.data = self.conv.bias.data.zero_() 
        #self.conv = nn.utils.weight_norm(self.conv)
        self.wscale_layer = WSConv2d(in_dim, out_dim, kernel_size, 1, padding)
        #self.conv = self.wscale_layer.conv
        self.output_dim = out_dim
        self.input_dim = in_dim

    def forward(self, x):
        x = self.wscale_layer(x)
        return x



def conv_bn_relu(in_dim, out_dim, kernel_size, padding=0):
    return nn.Sequential(
        EqualizedConv2D(in_dim, out_dim, kernel_size, padding),
        nn.LeakyReLU(negative_slope=.2),
        PixelwiseNormalization()
        #nn.BatchNorm2d(out_dim)
    )


def get_transition_value(x_old, x_new, transition_variable):
    return (1-transition_variable) * x_old + transition_variable*x_new

class PixelwiseNormalization(nn.Module):

    def __init__(self):
        super(PixelwiseNormalization, self).__init__()

    def forward(self, x):
        factor = ((x**2 ).mean(dim=1, keepdim=True)+ 1e-8)**0.5
        return x / factor

class UpSamplingBlock(nn.Module):
    def __init__(self):
        super(UpSamplingBlock, self).__init__()
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2)

class UnetDownSamplingBlock(nn.Module):

  def __init__(self, in_dim, out_dim, first_block):
    super(UnetDownSamplingBlock, self).__init__()

    modules = [
      conv_bn_relu(in_dim, out_dim, 3, 1),
    ]
    if not first_block:
      modules = [nn.MaxPool2d(2, 2)] + modules
    self.model = nn.Sequential(*modules)
  
  def forward(self, x):
    return self.model(x)

class UnetUpsamplingBlock(nn.Module):
  
  def __init__(self, in_dim, out_dim, first_block):
    super(UnetUpsamplingBlock, self).__init__()

    modules = [
      conv_bn_relu(in_dim, out_dim, 3, 1),
      conv_bn_relu(out_dim, out_dim, 3, 1) # This is wrong for the last block
    ]
    self.upsampling_block = UpSamplingBlock()
    
    self.model = nn.Sequential(*modules)
    self.first_block = first_block
  
  def forward(self, x, skip_x):
    if not self.first_block:
      x = self.upsampling_block(x)
      assert x.shape == skip_x.shape, "Shape of X: {}, skip shape: {}".format(x.shape, skip_x.shape)
      x += skip_x
    x = self.model(x)
    return x


class Generator(nn.Module):
  
    def __init__(self, 
                 noise_dim,
                 start_channel_dim,
                 image_channels):
        super(Generator, self).__init__()
        # Transition blockss
        self.image_channels = image_channels
        self.noise_dim = noise_dim
        self.to_rgb_new = EqualizedConv2D(start_channel_dim, self.image_channels, 1, 0)
        self.to_rgb_old = EqualizedConv2D(start_channel_dim, self.image_channels, 1, 0)
        self.down_sampling_blocks = [
          UnetDownSamplingBlock(image_channels, start_channel_dim, True).cuda(),
          UnetDownSamplingBlock(start_channel_dim, start_channel_dim, False),
          UnetDownSamplingBlock(start_channel_dim, start_channel_dim, False),
          UnetDownSamplingBlock(start_channel_dim, start_channel_dim, False),
        ]
        
        self.upsampling_blocks = [
          UnetUpsamplingBlock(start_channel_dim, start_channel_dim, True),
          UnetUpsamplingBlock(start_channel_dim, start_channel_dim, False),
          UnetUpsamplingBlock(start_channel_dim, start_channel_dim, False),
          UnetUpsamplingBlock(start_channel_dim, image_channels, False),
        ]
        for block in self.down_sampling_blocks + self.upsampling_blocks:
          to_cuda(block)
    
    def extend(self, output_dim):
      return
        #self.to_rgb_new.apply(init_weights)


    # x: Bx1x1x512
    def forward(self, x_in):
      downsampling_outputs = [x_in]
      for block in self.down_sampling_blocks:
        #print("en")
        x = block(downsampling_outputs[-1])
        downsampling_outputs += [x]
      #print([x.shape for x in downsampling_outputs])
      #print(len(self.upsampling_blocks))
      for i, block in enumerate(self.upsampling_blocks):
        #print(i)
        x = block(x, downsampling_outputs[-i-1])
        #print(x.shape)
      assert x.shape == x_in.shape
      return x + x_in
    
    def summary(self):
        print("="*80)
        print("GENERATOR")
        summary(self, (self.noise_dim, 1, 1))


def conv_module(dim_in, dim_out, kernel_size, padding, image_width):
    return nn.Sequential(
        EqualizedConv2D(dim_in, dim_out, kernel_size, padding),
        nn.LeakyReLU(negative_slope=.2)
    )

class Discriminator(nn.Module):

    def __init__(self, 
                 in_channels,
                 imsize,
                 start_channel_dim,
                 label_size
                 ):
        super(Discriminator, self).__init__()
        self.label_size = label_size
        self.image_channels = in_channels
        self.current_input_imsize = 4
        self.from_rgb_new = conv_module(in_channels, start_channel_dim, 1, 0, self.current_input_imsize)

        self.from_rgb_old = conv_module(in_channels, start_channel_dim, 1, 0, self.current_input_imsize)
        self.new_block = nn.Sequential()
        self.core_model = nn.Sequential(
            MinibatchStdLayer(),
            conv_module(start_channel_dim+1, start_channel_dim, 3, 1, imsize),
            conv_module(start_channel_dim, start_channel_dim, 4, 0, 1),            
        )
        self.output_layer = nn.Linear(start_channel_dim, 1 + label_size)
        self.transition_value = 1.0

        
    def extend(self, input_dim):
        
        self.current_input_imsize *= 2
        output_dim = next(self.from_rgb_new.children()).output_dim
        #output_dim = list(self.from_rgb_new.parameters())[1].shape[0]
        new_core_model = nn.Sequential()
        idx = 0
        for module in self.new_block.children():
            new_core_model.add_module(str(idx), module)
            idx += 1
        for module in self.core_model.children():
            new_core_model.add_module(str(idx), module)
            idx += 1
        self.core_model = new_core_model
        self.from_rgb_old = nn.Sequential(
            nn.AvgPool2d([2,2]),
            self.from_rgb_new
        )
        self.from_rgb_new = conv_module(self.image_channels, input_dim, 1, 0,self.current_input_imsize)
        self.from_rgb_new = to_cuda(self.from_rgb_new)
        self.new_block = nn.Sequential(
            conv_module(input_dim, input_dim, 3, 1, self.current_input_imsize),
            conv_module(input_dim, output_dim, 3, 1, self.current_input_imsize),
            nn.AvgPool2d([2, 2])
        )
        self.new_block = to_cuda(self.new_block)
        #self.new_block.apply(init_weights)
        #self.from_rgb_new.apply(init_weights)



    # x: Bx1x1x512
    def forward(self, x):
        x_old = self.from_rgb_old(x)
        x_new = self.from_rgb_new(x)
        x_new = self.new_block(x_new)
        x = get_transition_value(x_old, x_new, self.transition_value)
        x = self.core_model(x)
        x = x.view(x.shape[0], -1)
        x = self.output_layer(x)
        if self.label_size > 0:
            return x[:, :1], x[:, 1:]
        return x, None
    
    def summary(self):
        print("="*80)
        print("GENERATOR")
        summary(self, (self.image_channels, self.current_input_imsize, self.current_input_imsize))


if __name__ == "__main__":

    # Test that logits is the same

    # real data
    z = to_cuda(torch.zeros((64, 128, 1,1)))
    d = Discriminator(1, 4).cuda()
    g = Generator(128).cuda()
    data = g(z, 1)
    logits = d(data, 1)


    print(logits)

    d.extend(32)
    g.extend(32)

    data = g(z, 0)
    lg = d(data, 0)
    print(lg)

