
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import to_cuda, init_weights
from torchsummary import summary
import numpy as np
from utils import get_transition_value
from custom_layers import PixelwiseNormalization, WSConv2d, WSLinear, MinibatchStdLayer, UpSamplingBlock

def conv_bn_relu(in_dim, out_dim, kernel_size, padding=0):
    return nn.Sequential(
        WSConv2d(in_dim, out_dim, kernel_size, padding),
        nn.LeakyReLU(negative_slope=.2),
        PixelwiseNormalization()
        #nn.BatchNorm2d(out_dim)
    )

class UnetDownSamplingBlock(nn.Module):

  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.model = nn.Sequential(
      conv_bn_relu(in_dim, out_dim, 3, 1),
      conv_bn_relu(out_dim, out_dim, 3, 1),
      #nn.Dropout2d(0.1)
    )  
  def forward(self, x):
    return self.model(x)

class UnetUpsamplingBlock(nn.Module):
  
  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.model = nn.Sequential(
      nn.Dropout2d(0.1),
      conv_bn_relu(in_dim, out_dim, 3, 1),
      conv_bn_relu(out_dim, out_dim, 3, 1) # This is wrong for the last block
    )
  
  def forward(self, x):
    x = self.model(x)
    return x


class Generator(nn.Module):
  
    def __init__(self, 
                 noise_dim,
                 start_channel_dim,
                 image_channels):
        super().__init__()
        # Transition blockss
        self.image_channels = image_channels
        self.noise_dim = noise_dim
        self.z_channel = noise_dim // 4 // 4
        self.transition_value = 1.0
        self.to_rgb_new = WSConv2d(start_channel_dim, self.image_channels, 1, 0)
        self.to_rgb_old = WSConv2d(start_channel_dim, self.image_channels, 1, 0)

        self.core_blocks_down = nn.ModuleList([
            to_cuda(UnetDownSamplingBlock(start_channel_dim, start_channel_dim)),
        ])
        self.core_blocks_up = nn.ModuleList([
            to_cuda(UnetUpsamplingBlock(start_channel_dim, start_channel_dim))
        ])
        self.new_up = nn.Sequential()
        self.old_up = nn.Sequential()
        self.new_down = nn.Sequential()
        self.from_rgb_new = to_cuda(conv_bn_relu(self.image_channels, start_channel_dim, 1, 0))
        self.from_rgb_old = to_cuda(conv_bn_relu(self.image_channels, start_channel_dim, 1, 0)) # Should be conv_bn_relu
        self.to_rgb = to_cuda(WSConv2d(self.image_channels*2, self.image_channels, 1, 0))
        self.current_imsize = 4
        self.upsampling = UpSamplingBlock()
        self.prev_channel_size = start_channel_dim
        self.downsampling = nn.AvgPool2d(2)

    
    def extend(self, output_dim):

        print("extending G")
        # Downsampling module
        self.current_imsize *= 2

        self.from_rgb_old = nn.Sequential(
            nn.AvgPool2d([2,2]),
            self.from_rgb_new
        )
        if self.current_imsize == 8:
            core_block_up = nn.Sequential(
                self.core_blocks_up[0],
                UpSamplingBlock()
            )
            self.core_blocks_up = nn.ModuleList([core_block_up])
        else:
            core_blocks_down = nn.ModuleList()

            core_blocks_down.append(self.new_down)
            first = [nn.AvgPool2d(2)] + list(self.core_blocks_down[0].children()) # Add donwsampling to downsampling block
            core_blocks_down.append(nn.Sequential(*first))
            core_blocks_down.extend(self.core_blocks_down[1:])

            self.core_blocks_down = core_blocks_down
            new_up_blocks = list(self.new_up.children()) + [UpSamplingBlock()]
            self.new_up = nn.Sequential(*new_up_blocks)
            self.core_blocks_up.append(self.new_up)

        self.from_rgb_new = to_cuda(conv_bn_relu(self.image_channels, output_dim, 1, 0))
        self.new_down = nn.Sequential(
            UnetDownSamplingBlock(output_dim, self.prev_channel_size)
        )
        self.new_down = to_cuda(self.new_down)
        # Upsampling modules
        self.to_rgb_old = self.to_rgb_new
        self.to_rgb_new = to_cuda(WSConv2d(output_dim, self.image_channels, 1, 0))

        self.new_up = to_cuda(nn.Sequential(
            UnetUpsamplingBlock(self.prev_channel_size*2, output_dim)
            
        ))
        self.prev_channel_size = output_dim

        

    # x: Bx1x1x512
    def forward(self, x_in, z, dropout_rate=0.5):
        unet_skips = []
        if self.current_imsize != 4:
            old_down = self.from_rgb_old(x_in)
            new_down = self.from_rgb_new(x_in)
            new_down = self.new_down(new_down) 
            unet_skips.append(new_down)
            new_down = self.downsampling(new_down)
            x = get_transition_value(old_down, new_down, self.transition_value)
        else:
            x = self.from_rgb_new(x_in)
        
        
        for block in self.core_blocks_down[:-1]:
            x = block(x)
            unet_skips.append(x)
        x = self.core_blocks_down[-1](x)
        x = self.core_blocks_up[0](x)

        for idx, block in enumerate(self.core_blocks_up[1:]):
            skip_x = unet_skips[-idx-1]
            assert skip_x.shape == x.shape, "IDX: {}, skip_x: {}, x: {}".format(idx, skip_x.shape, x.shape)
            x = torch.cat((x, skip_x), dim=1)
            x = block(x)
        
        if self.current_imsize == 4:
            x = self.to_rgb_new(x)
            return x
        x_old = self.to_rgb_old(x)
        x = torch.cat((x, unet_skips[0]), dim=1)
        x_new = self.new_up(x)
        x_new = self.to_rgb_new(x_new)
        x = get_transition_value(x_old, x_new, self.transition_value)
        return x

    
    def summary(self):
        print("="*80)
        print("GENERATOR")
        #summary(self, (self.image_channels, self.current_imsize, self.current_imsize))


def conv_module_bn(dim_in, dim_out, kernel_size, padding):
    return nn.Sequential(
        WSConv2d(dim_in, dim_out, kernel_size, padding),
        nn.LeakyReLU(negative_slope=.2)
    )


class Discriminator(nn.Module):

    def __init__(self, 
                 in_channels,
                 imsize,
                 start_channel_dim
                 ):
        super(Discriminator, self).__init__()
        self.image_channels = in_channels
        self.current_input_imsize = 4
        self.from_rgb_new = conv_module_bn(in_channels*2, start_channel_dim, 1, 0)

        self.from_rgb_old = conv_module_bn(in_channels*2, start_channel_dim, 1, 0)
        self.new_block = nn.Sequential()
        self.core_model = nn.Sequential(
            MinibatchStdLayer(),
            conv_module_bn(start_channel_dim+1, start_channel_dim, 3, 1),
            conv_module_bn(start_channel_dim, start_channel_dim, 4, 0),            
        )
        self.output_layer = WSLinear(start_channel_dim, 1)
        self.transition_value = 1.0
        self.prev_channel_dim = start_channel_dim

        
    def extend(self, input_dim):
        
        self.current_input_imsize *= 2
        output_dim = self.prev_channel_dim
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
        self.from_rgb_new = conv_module_bn(self.image_channels*2, input_dim, 1, 0)
        self.from_rgb_new = to_cuda(self.from_rgb_new)
        self.new_block = nn.Sequential(
            conv_module_bn(input_dim, input_dim, 3, 1),
            conv_module_bn(input_dim, output_dim, 3, 1),
            nn.AvgPool2d([2, 2])
        )
        self.new_block = to_cuda(self.new_block)
        self.prev_channel_dim = input_dim



    # x: Bx1x1x512
    def forward(self, x, condition):
        x = torch.cat((x, condition), dim=1)
        x_old = self.from_rgb_old(x)
        x_new = self.from_rgb_new(x)
        x_new = self.new_block(x_new)
        x = get_transition_value(x_old, x_new, self.transition_value)
        self.from_rgb = x
        x = self.core_model(x)
        self.after_core = x
        x = x.view(x.shape[0], -1)
        x = self.output_layer(x)
        return x, None
    
    def summary(self):
        print("="*80)
        print("DISCRIMINATOR")
        #summary(self, (self.image_channels, self.current_input_imsize, self.current_input_imsize))


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

