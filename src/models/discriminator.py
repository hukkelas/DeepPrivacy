import torch
import torch.nn as nn
from utils import to_cuda
from utils import get_transition_value
from models.custom_layers import WSConv2d, WSLinear
from models.utils import generate_pose_channel_images


def conv_module_bn(dim_in, dim_out, kernel_size, padding):
    return nn.Sequential(
        WSConv2d(dim_in, dim_out, kernel_size, padding),
        nn.LeakyReLU(negative_slope=.2)
    )


class ResNetBlock(nn.Module):

    def __init__(self, num_channels, num_conv):
        super(ResNetBlock, self).__init__()
        blocks = []
        for i in range(num_conv):
            blocks.append(
           conv_module_bn(num_channels, num_channels, 3, 1)
            )
        self.conv = nn.Sequential(*blocks)

    def forward(self, x):
        prev = x
        block_out = self.conv(x)
        res = (prev + block_out)/2
        return res


class Discriminator(nn.Module):

    def __init__(self, 
                 in_channels,
                 start_channel_dim,
                 pose_size
                 ):
        self.orig_start_channel_dim = start_channel_dim
        start_channel_dim = int(start_channel_dim*(2**0.5))
        start_channel_dim = start_channel_dim // 8 * 8
        super(Discriminator, self).__init__()
        
        self.num_poses = pose_size // 2
        self.image_channels = in_channels
        self.current_imsize = 4
        self.from_rgb_new = conv_module_bn(in_channels*2, start_channel_dim, 1, 0)

        self.from_rgb_old = conv_module_bn(in_channels*2, start_channel_dim, 1, 0)
        self.new_block = nn.Sequential()
        self.core_model = nn.Sequential(
            nn.Sequential(
                conv_module_bn(start_channel_dim + self.num_poses, start_channel_dim, 3, 1),
                conv_module_bn(start_channel_dim, start_channel_dim, 4, 0),
            )
        )
        self.core_model = to_cuda(self.core_model)
        self.output_layer = WSLinear(start_channel_dim, 1)
        self.transition_value = 1.0
        self.prev_channel_dim = start_channel_dim
        self.extension_channels = []

    def extend(self, input_dim):
        self.extension_channels.append(input_dim)
        input_dim = int(input_dim * (2**0.5))
        input_dim = input_dim//8 * 8 
        self.current_imsize *= 2
        output_dim = self.prev_channel_dim
        if not self.current_imsize == 8:
            self.core_model = nn.Sequential(
                self.new_block,
                *self.core_model.children()
            )
        self.from_rgb_old = nn.Sequential(
            nn.AvgPool2d([2, 2]),
            self.from_rgb_new
        )
        self.from_rgb_new = conv_module_bn(self.image_channels*2, input_dim, 1, 0)
        self.from_rgb_new = to_cuda(self.from_rgb_new)
        self.new_block = nn.Sequential(
            conv_module_bn(input_dim+self.num_poses, input_dim, 3, 1),
            conv_module_bn(input_dim, output_dim, 3, 1),
            nn.AvgPool2d([2, 2])
        )
        self.new_block = to_cuda(self.new_block)
        self.prev_channel_dim = input_dim

    def forward(self, x, condition, pose_info):
        pose_channels = generate_pose_channel_images(4,
                                                     self.current_imsize,
                                                     x.device,
                                                     pose_info,
                                                     x.dtype)
        x = torch.cat((x, condition), dim=1)
        x_old = self.from_rgb_old(x)
        x_new = self.from_rgb_new(x)
        if self.current_imsize != 4:
            x_new = torch.cat((x_new, pose_channels[-1]), dim=1)
        x_new = self.new_block(x_new)
        x = get_transition_value(x_old, x_new, self.transition_value)
        idx = 1 if self.current_imsize == 4 else 2
        for block in self.core_model.children():
            x = torch.cat((x, pose_channels[-idx]), dim=1)
            idx += 1
            x = block(x)

        x = x.view(x.shape[0], -1)
        x = self.output_layer(x)
        return x


class DeepDiscriminator(nn.Module):

    def __init__(self, 
                 in_channels,
                 start_channel_dim,
                 pose_size
                 ):
        self.orig_start_channel_dim = start_channel_dim
        #start_channel_dim = int(start_channel_dim*(2**0.5))
        super(DeepDiscriminator, self).__init__()
        
        self.num_poses = pose_size // 2
        self.image_channels = in_channels
        self.current_imsize = 4
        self.from_rgb_new = conv_module_bn(in_channels*2, start_channel_dim, 1, 0)

        self.from_rgb_old = conv_module_bn(in_channels*2, start_channel_dim, 1, 0)
        self.new_block = nn.Sequential()
        self.core_model = nn.Sequential(
            nn.Sequential(
                conv_module_bn(start_channel_dim + self.num_poses, start_channel_dim, 1, 0),
                ResNetBlock(start_channel_dim, 3),
                conv_module_bn(start_channel_dim, start_channel_dim, 4, 0)
            )
        )
        self.core_model = to_cuda(self.core_model)
        self.output_layer = WSLinear(start_channel_dim, 1)
        self.transition_value = 1.0
        self.prev_channel_dim = start_channel_dim
        self.extension_channels = []

    def extend(self, input_dim):
        self.extension_channels.append(input_dim)
        #input_dim = int(input_dim * (2**0.5))
        self.current_imsize *= 2
        output_dim = self.prev_channel_dim
        if not self.current_imsize == 8:
            self.core_model = nn.Sequential(
                self.new_block,
                *self.core_model.children()
            )
        self.from_rgb_old = nn.Sequential(
            nn.AvgPool2d([2, 2]),
            self.from_rgb_new
        )
        self.from_rgb_new = conv_module_bn(self.image_channels*2, input_dim, 1, 0)
        self.from_rgb_new = to_cuda(self.from_rgb_new)
        self.new_block = nn.Sequential(
            conv_module_bn(input_dim + self.num_poses, input_dim, 1, 0),
            ResNetBlock(input_dim, 4),
            conv_module_bn(input_dim, output_dim, 1, 0),
            nn.AvgPool2d([2, 2])
        )
        self.new_block = to_cuda(self.new_block)
        self.prev_channel_dim = input_dim

    def forward(self, x, condition, pose_info):
        pose_channels = generate_pose_channel_images(4,
                                                     self.current_imsize,
                                                     x.device,
                                                     pose_info,
                                                     x.dtype)
        x = torch.cat((x, condition), dim=1)
        x_old = self.from_rgb_old(x)
        x_new = self.from_rgb_new(x)
        if self.current_imsize != 4:
            x_new = torch.cat((x_new, pose_channels[-1]), dim=1)
        x_new = self.new_block(x_new)
        x = get_transition_value(x_old, x_new, self.transition_value)
        idx = 1 if self.current_imsize == 4 else 2
        for block in self.core_model.children():
            x = torch.cat((x, pose_channels[-idx]), dim=1)
            idx += 1
            x = block(x)

        x = x.view(x.shape[0], -1)
        x = self.output_layer(x)
        return x
