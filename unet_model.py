
import torch
import torch.nn as nn
from utils import to_cuda
from utils import get_transition_value
from custom_layers import PixelwiseNormalization, WSConv2d, WSLinear, UpSamplingBlock


def conv_bn_relu(in_dim, out_dim, kernel_size, padding=0):
    return nn.Sequential(
        WSConv2d(in_dim, out_dim, kernel_size, padding),
        nn.LeakyReLU(negative_slope=.2),
        PixelwiseNormalization()
    )


class UnetDownSamplingBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.Sequential(
            conv_bn_relu(in_dim, out_dim, 3, 1),
            conv_bn_relu(out_dim, out_dim, 3, 1),
        )

    def forward(self, x):
        return self.model(x)


class UnetUpsamplingBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.Sequential(
            conv_bn_relu(in_dim, out_dim, 3, 1),
            conv_bn_relu(out_dim, out_dim, 3, 1)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def generate_pose_channel_images(min_imsize, max_imsize, device, pose_information):
    batch_size = pose_information.shape[0]
    pose_x = pose_information[:, range(0, pose_information.shape[1], 2)].view(-1)
    pose_y = pose_information[:, range(1, pose_information.shape[1], 2)].view(-1)
    legal_mask = ((pose_x < 0 ) + (pose_x >= 1.0) + (pose_y <0) + (pose_y >= 1.0)) == 0
    batch_idx = torch.cat([torch.ones(pose_information.shape[1]//2)*k for k in range(batch_size)]).long()

    pose_x = pose_x[legal_mask]
    pose_y = pose_y[legal_mask]
    batch_idx = batch_idx[legal_mask]


    pose_images = []
    imsize = min_imsize
    while imsize <= max_imsize:
        new_im = torch.zeros((batch_size, 1, imsize, imsize))
        px = (pose_x * imsize).long()
        py = (pose_y * imsize).long()
        new_im[batch_idx, 0, py, px] = 1
        new_im = new_im.to(device)
        pose_images.append(new_im)
        imsize *= 2
    return pose_images


class Generator(nn.Module):

    def __init__(self,
                 pose_size,
                 start_channel_dim,
                 image_channels):
        super().__init__()
        # Transition blockss
        self.image_channels = image_channels
        self.transition_value = 1.0
        self.to_rgb_new = WSConv2d(start_channel_dim, self.image_channels, 1, 0)
        self.to_rgb_old = WSConv2d(start_channel_dim, self.image_channels, 1, 0)

        self.core_blocks_down = nn.ModuleList([
            to_cuda(UnetDownSamplingBlock(start_channel_dim, start_channel_dim)),
        ])
        self.core_blocks_up = nn.ModuleList([
            to_cuda(UnetUpsamplingBlock(start_channel_dim+1, start_channel_dim))
        ])

        self.new_up = nn.Sequential()
        self.old_up = nn.Sequential()
        self.new_down = nn.Sequential()
        self.from_rgb_new = to_cuda(
            conv_bn_relu(self.image_channels, start_channel_dim, 1, 0))
        self.from_rgb_old = to_cuda(
            conv_bn_relu(self.image_channels, start_channel_dim, 1, 0))
        self.current_imsize = 4
        self.upsampling = UpSamplingBlock()
        self.prev_channel_size = start_channel_dim
        self.downsampling = nn.AvgPool2d(2)

    def extend(self, output_dim):
        print("extending G", output_dim)
        # Downsampling module
        self.current_imsize *= 2

        self.from_rgb_old = nn.Sequential(
            nn.AvgPool2d([2, 2]),
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
            first = [nn.AvgPool2d(2)] + list(self.core_blocks_down[0].children())
            core_blocks_down.append(nn.Sequential(*first))
            core_blocks_down.extend(self.core_blocks_down[1:])

            self.core_blocks_down = core_blocks_down
            new_up_blocks = list(self.new_up.children()) + [UpSamplingBlock()]
            self.new_up = nn.Sequential(*new_up_blocks)
            self.core_blocks_up.append(self.new_up)

        self.from_rgb_new = to_cuda(
            conv_bn_relu(self.image_channels, output_dim, 1, 0))
        self.new_down = nn.Sequential(
            UnetDownSamplingBlock(output_dim, self.prev_channel_size)
        )
        self.new_down = to_cuda(self.new_down)
        # Upsampling modules
        self.to_rgb_old = self.to_rgb_new
        self.to_rgb_new = to_cuda(
            WSConv2d(output_dim, self.image_channels, 1, 0))

        self.new_up = to_cuda(nn.Sequential(
            UnetUpsamplingBlock(self.prev_channel_size*2+1, output_dim)
        ))
        self.prev_channel_size = output_dim

    def forward(self, x_in, pose_info):
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

        pose_channels = generate_pose_channel_images(4,
                                                     self.current_imsize, 
                                                     x_in.device,
                                                     pose_info)
        x = torch.cat((x, pose_channels[0]), dim=1)

        x = self.core_blocks_up[0](x)

        for idx, block in enumerate(self.core_blocks_up[1:]):
            skip_x = unet_skips[-idx-1]
            assert skip_x.shape == x.shape, "IDX: {}, skip_x: {}, x: {}".format(idx, skip_x.shape, x.shape)
            x = torch.cat((x, skip_x, pose_channels[idx+1]), dim=1)
            x = block(x)

        if self.current_imsize == 4:
            x = self.to_rgb_new(x)
            return x
        x_old = self.to_rgb_old(x)
        x = torch.cat((x, unet_skips[0], pose_channels[-1]), dim=1)
        x_new = self.new_up(x)
        x_new = self.to_rgb_new(x_new)
        x = get_transition_value(x_old, x_new, self.transition_value)
        return x


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
            nn.Sequential(
                conv_module_bn(start_channel_dim+1, start_channel_dim, 3, 1),
                conv_module_bn(start_channel_dim, start_channel_dim, 4, 0),
            )
        )
        self.output_layer = WSLinear(start_channel_dim, 1)
        self.transition_value = 1.0
        self.prev_channel_dim = start_channel_dim

    def extend(self, input_dim):
        self.current_input_imsize *= 2
        output_dim = self.prev_channel_dim
        if not self.current_input_imsize == 8:
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
            conv_module_bn(input_dim+1, input_dim, 3, 1),
            conv_module_bn(input_dim, output_dim, 3, 1),
            nn.AvgPool2d([2, 2])
        )
        self.new_block = to_cuda(self.new_block)
        self.prev_channel_dim = input_dim

    def forward(self, x, condition, pose_info):
        pose_channels = generate_pose_channel_images(4,
                                                     self.current_input_imsize,
                                                     x.device,
                                                     pose_info)
        x = torch.cat((x, condition), dim=1)
        x_old = self.from_rgb_old(x)
        x_new = self.from_rgb_new(x)
        if self.current_input_imsize != 4:
            x_new = torch.cat((x_new, pose_channels[-1]), dim=1)
        x_new = self.new_block(x_new)
        x = get_transition_value(x_old, x_new, self.transition_value)
        idx = 1 if self.current_input_imsize == 4 else 2
        for block in self.core_model.children():
            x = torch.cat((x, pose_channels[-idx]), dim=1)
            idx += 1
            x = block(x)

        x = x.view(x.shape[0], -1)
        x = self.output_layer(x)
        return x
