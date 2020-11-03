import torch
import numpy as np
import torch.nn as nn
from ..base import ProgressiveBase, FromRGB
from .. import layers, blocks
from ..utils import generate_pose_channel_images
from ..build import DISCRIMINATOR_REGISTRY


def get_conv_size(cfg, size):
    size = size * (2**0.5)
    size = size * cfg.models.discriminator.conv_multiplier
    if cfg.models.generator.conv2d_config.conv.type == "gconv":
        size *= (2**0.5)
    return int(np.ceil(size / 8) * 8)


class FromRGB(FromRGB):

    def conv_channel_size(self):
        size = super().conv_channel_size()
        return get_conv_size(self.cfg, size)

    def prev_conv_channel_size(self):
        size = super().conv_channel_size()
        return get_conv_size(self.cfg, size)


@DISCRIMINATOR_REGISTRY.register_module
class Discriminator(ProgressiveBase):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg)
        self.min_fmap_resolution = cfg.models.discriminator.min_fmap_resolution
        self.current_imsize = self.min_fmap_resolution
        self._use_pose = self.cfg.models.pose_size > 0
        self._one_hot_pose = (
            self._use_pose and
            not self.cfg.models.discriminator.scalar_pose_input)
        self.layers = nn.Sequential()
        self.layers.add_module(
            "from_rgb", FromRGB(
                cfg, cfg.models.discriminator.conv2d_config,
                in_channels=cfg.models.image_channels * 2 + 1,
                current_imsize=self.min_fmap_resolution))
        if self._one_hot_pose:
            self.layers.add_module(
                "pose_concat0", layers.OneHotPoseConcat())

        # Last convolution has kernel size (4, 4)
        first_block = self.build_block()
        layers_ = list(first_block.layers)
        last_conv_idx = [
            i for i, x in enumerate(layers_)
            if isinstance(x, layers.Conv2d)][-1]
        layers_[last_conv_idx] = blocks.build_base_conv(
            cfg.models.discriminator.conv2d_config,
            True,
            first_block.out_channels[-2],
            first_block.out_channels[-1],
            kernel_size=(4, 4), padding=0)
        first_block.layers = nn.Sequential(*layers_)
        self.layers.add_module("basic_block0", first_block)
        num_outputs = 1
        res = self.min_fmap_resolution - 3
        self.output_layer = layers.Linear(
            self.conv_channel_size() * res**2, num_outputs)

    def conv_channel_size(self):
        size = super().conv_channel_size()
        return get_conv_size(self.cfg, size)

    def prev_conv_channel_size(self):
        size = super().prev_conv_channel_size()
        return get_conv_size(self.cfg, size)

    def build_block(self):
        end_size = self.conv_channel_size()
        if self.current_imsize != 4:
            end_size = self.prev_conv_channel_size()
        start_size = self.conv_channel_size()
        if self._use_pose:
            pose_imsize = self.cfg.models.discriminator.scalar_pose_input_imsize
            if self._one_hot_pose:
                start_size += self.cfg.models.pose_size // 2
            elif pose_imsize == self.current_imsize:
                start_size += 1
        # No residual layer for last block.
        residual = self.cfg.models.discriminator.residual and self.current_imsize != self.min_fmap_resolution
        return blocks.BasicBlock(
            self.cfg.models.discriminator.conv2d_config,
            self.current_imsize, start_size, [start_size, end_size],
            residual=residual
        )

    def extend(self):
        super().extend()
        from_rgb, *_layers = list(self.layers.children())
        from_rgb.extend()
        _layers = nn.Sequential()
        _layers.add_module("from_rgb", from_rgb)
        i = self.transition_step
        pose_imsize = self.cfg.models.discriminator.scalar_pose_input_imsize
        if self._one_hot_pose:
            _layers.add_module(f"pose_concat{i}", layers.OneHotPoseConcat())
        elif self._use_pose and pose_imsize == self.current_imsize:
            output_shape = (1, pose_imsize, pose_imsize)
            pose_fcnn = blocks.ScalarPoseFCNN(
                self.cfg.models.pose_size, 64, output_shape)
            _layers.add_module("pose_fcnn", pose_fcnn)

        _layers.add_module(f"basic_block{i}", self.build_block())
        _layers.add_module(
            f"downsample{i}", layers.AvgPool2d(kernel_size=2))
        if self.progressive_enabled:
            _layers.add_module("transition_block", layers.TransitionBlock())
        for name, module in self.layers.named_children():
            if isinstance(module, FromRGB):
                continue
            if isinstance(module, layers.TransitionBlock):
                continue
            _layers.add_module(name, module)
        self.layers = _layers

    def forward_fake(self, condition, mask, landmarks=None,
                     fake_img=None, with_pose=False, **kwargs):
        return self(
            fake_img, condition, mask, landmarks, with_pose=with_pose
        )

    def forward(
            self, img, condition, mask, landmarks=None,
            with_pose=False, **kwargs):
        landmarks_oh = None
        if self._one_hot_pose:
            landmarks_oh = generate_pose_channel_images(
                4, self.current_imsize, condition.device, landmarks,
                condition.dtype)
        batch = dict(
            landmarks_oh=landmarks_oh,
            landmarks=landmarks,
            transition_value=self.transition_value)
        x = torch.cat((img, condition, mask), dim=1)
        x, mask, batch = self.layers((x, mask, batch))
        x = x.view(x.shape[0], -1)
        x = self.output_layer(x)
        return [x]


if __name__ == "__main__":
    from deep_privacy.config import Config, default_parser
    args = default_parser().parse_args()
    cfg = Config.fromfile(args.config_path)

    g = Discriminator(cfg).cuda()
    [g.extend() for i in range(3)]
    g.cuda()
    print(g)
    imsize = g.current_imsize
    batch = dict(
        img=torch.randn((8, 3, imsize, imsize)).cuda(),
        mask=torch.ones((8, 1, imsize, imsize)).cuda(),
        condition=torch.randn((8, 3, imsize, imsize)).cuda(),
        landmarks=torch.randn((8, 14)).cuda()
    )
    print(g(**batch).shape)
