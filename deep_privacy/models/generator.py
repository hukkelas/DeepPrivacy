import torch
import torch.nn as nn
from deep_privacy.models.custom_layers import PixelwiseNormalization, WSConv2d, UpSamplingBlock
from deep_privacy.models.utils import generate_pose_channel_images, get_transition_value
from deep_privacy.models.base_model import ProgressiveBaseModel

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

class Generator(ProgressiveBaseModel):

    def __init__(self,
                 pose_size,
                 start_channel_dim,
                 image_channels):
        super().__init__(pose_size, start_channel_dim, image_channels)
        # Transition blockss
        self.orig_start_channel_dim  = start_channel_dim
        self.to_rgb_new = WSConv2d(start_channel_dim, self.image_channels, 1, 0)
        self.to_rgb_old = WSConv2d(start_channel_dim, self.image_channels, 1, 0)

        self.core_blocks_down = nn.ModuleList([
            UnetDownSamplingBlock(start_channel_dim, start_channel_dim)
        ])
        self.core_blocks_up = nn.ModuleList([
            nn.Sequential(
                conv_bn_relu(start_channel_dim+self.num_poses+32, start_channel_dim, 1, 0),
                UnetUpsamplingBlock(start_channel_dim, start_channel_dim)
            )
        ])

        self.new_up = nn.Sequential()
        self.old_up = nn.Sequential()
        self.new_down = nn.Sequential()
        self.from_rgb_new = conv_bn_relu(self.image_channels, start_channel_dim, 1, 0)
        self.from_rgb_old =  conv_bn_relu(self.image_channels, start_channel_dim, 1, 0)
        
        self.upsampling = UpSamplingBlock()
        self.downsampling = nn.AvgPool2d(2)

    def extend(self):
        output_dim = self.transition_channels[self.transition_step]
        print("extending G", output_dim)
        # Downsampling module

        self.from_rgb_old = nn.Sequential(
            nn.AvgPool2d([2, 2]),
            self.from_rgb_new
        )
        if self.transition_step == 0:
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

        self.from_rgb_new =  conv_bn_relu(self.image_channels, output_dim, 1, 0)
        self.new_down = nn.Sequential(
            UnetDownSamplingBlock(output_dim, self.prev_channel_extension)
        )
        self.new_down = self.new_down
        # Upsampling modules
        self.to_rgb_old = self.to_rgb_new
        self.to_rgb_new = WSConv2d(output_dim, self.image_channels, 1, 0)

        self.new_up = nn.Sequential(
            conv_bn_relu(self.prev_channel_extension*2+self.num_poses, self.prev_channel_extension, 1, 0),
            UnetUpsamplingBlock(self.prev_channel_extension, output_dim)
        )
        super().extend()

    def new_parameters(self):
        new_paramters = list(self.new_down.parameters()) + list(self.to_rgb_new.parameters())
        new_paramters += list(self.new_up.parameters()) + list(self.from_rgb_new.parameters())
        return new_paramters

    def generate_latent_variable(self, *args):
        if len(args) == 1:
            x_in = args[0]
            return torch.randn(x_in.shape[0], 32, 4, 4,
                               device=x_in.device,
                               dtype=x_in.dtype)
        elif len(args) == 3:
            batch_size, device, dtype = args
            return torch.randn(batch_size, 32, 4, 4,
                               device=device,
                               dtype=dtype)
        raise ValueError(f"Expected either x_in or (batch_size, device, dtype. Got: {args}")

    def forward(self, x_in, pose_info, z=None):
        if z is None:
            z = self.generate_latent_variable(x_in)
        unet_skips = []
        if self.transition_step != 0:
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
                                                     pose_info,
                                                     x_in.dtype)
        x = torch.cat((x, pose_channels[0], z), dim=1)
        x = self.core_blocks_up[0](x)

        for idx, block in enumerate(self.core_blocks_up[1:]):
            skip_x = unet_skips[-idx-1]
            assert skip_x.shape == x.shape, "IDX: {}, skip_x: {}, x: {}".format(idx, skip_x.shape, x.shape)
            x = torch.cat((x, skip_x, pose_channels[idx+1]), dim=1)
            x = block(x)

        if self.transition_step == 0:
            x = self.to_rgb_new(x)
            return x
        x_old = self.to_rgb_old(x)
        x = torch.cat((x, unet_skips[0], pose_channels[-1]), dim=1)
        x_new = self.new_up(x)
        x_new = self.to_rgb_new(x_new)
        x = get_transition_value(x_old, x_new, self.transition_value)
        return x