import torch.nn as nn
import torch
from .. import layers, blocks
from ..build import GENERATOR_REGISTRY
from .base import RunningAverageGenerator
from .gblocks import LatentVariableConcat, UnetSkipConnection
from ..base import ProgressiveBase, FromRGB
from ..utils import generate_pose_channel_images


class DecoderUpsample(layers.NearestUpsample):

    def forward(self, _inp):
        x_old, mask_old, batch = _inp
        x, mask = super().forward((x_old, mask_old))
        batch["x_old"] = x
        batch["mask_old"] = mask
        return x, mask, batch


class ToRGB(ProgressiveBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.current_imsize = cfg.models.generator.min_fmap_resolution
        self.min_fmap_resolution = self.current_imsize
        self.conv = blocks.build_base_conv(
            cfg.models.generator.conv2d_config,
            post_act=False,
            in_channels=self.conv_channel_size(),
            out_channels=cfg.models.image_channels,
            kernel_size=1, padding=0
        )
        self.old_conv = nn.Sequential()

    def extend(self):
        super().extend()
        self.old_conv = self.conv
        self.conv = blocks.build_base_conv(
            self.cfg.models.generator.conv2d_config,
            post_act=False,
            in_channels=self.conv_channel_size(),
            out_channels=self.cfg.models.image_channels,
            kernel_size=1, padding=0
        )

    def forward(self, _inp):
        x, mask, batch = _inp
        x, mask = self.conv((x, mask))
        if not self.progressive_enabled:
            return (x, mask, batch)
        x_old, mask_old = batch["x_old"], batch["mask_old"]
        x_old, mask_old = self.old_conv((x_old, mask_old))
        batch["x_old"] = x_old
        batch["mask_old"] = mask_old
        return x, mask, batch


@GENERATOR_REGISTRY.register_module
class Generator(RunningAverageGenerator, ProgressiveBase):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg=cfg, *args, **kwargs)
        self.current_imsize = cfg.models.generator.min_fmap_resolution
        self.min_fmap_resolution = self.current_imsize
        # Attributes
        conv2d_config = self.cfg.models.generator.conv2d_config
        self.concat_input_mask = conv2d_config.conv.type in ["conv", "gconv"]

        self._one_hot_pose = (
            not self.cfg.models.generator.scalar_pose_input
            and self.cfg.models.pose_size > 0)
        self._init_decoder()
        self._init_encoder()

    def _init_encoder(self):
        self.encoder = nn.ModuleList([
        ])
        frgb = FromRGB(
            self.cfg, self.cfg.models.generator.conv2d_config,
            in_channels=self.cfg.models.image_channels + self.concat_input_mask * 2,
            current_imsize=self.current_imsize)
        self.encoder.add_module("from_rgb", frgb)
        self.encoder.add_module(
            "basic_block0", blocks.BasicBlock(
                self.cfg.models.generator.conv2d_config, self.current_imsize,
                self.conv_channel_size(),
                [self.conv_channel_size(), self.conv_channel_size()],
                residual=self.cfg.models.generator.residual)
        )

    def _init_decoder(self):
        self.decoder = nn.ModuleList([])
        self.decoder.add_module(
            "latent_variable_concat", LatentVariableConcat(self.conv2d_config)
        )
        if self.cfg.models.generator.scalar_pose_input:
            m = self.min_fmap_resolution
            pose_shape = (16, m, m)
            self.decoder.add_module(
                "pose_fcnn", blocks.ScalarPoseFCNN(
                    self.cfg.models.pose_size, 128, pose_shape))
        elif self.cfg.models.pose_size != 0:
            self.decoder.add_module("pose_concat0", layers.OneHotPoseConcat())
        self.decoder.add_module(
            "basic_block0", self.create_up_block(
                resolution=self.min_fmap_resolution))
        self.decoder.add_module("to_rgb", ToRGB(self.cfg))

    def extend_decoder(self):
        to_rgb = [_ for _ in self.decoder if isinstance(_, ToRGB)][0]
        to_rgb.extend()
        decoder = nn.ModuleList([])
        for name, module in self.decoder.named_children():
            if isinstance(
                    module, ToRGB) or isinstance(
                    module, layers.TransitionBlock):
                continue
            decoder.add_module(name, module)
        self.decoder = decoder
        i = self.transition_step
        self.decoder.add_module(f"upsample{i}", DecoderUpsample())
        if self.cfg.models.generator.unet.enabled:
            self.decoder.add_module(
                f"skip_connection{i}", UnetSkipConnection(
                    self.conv2d_config, self.prev_conv_channel_size() * 2,
                    self.prev_conv_channel_size(),
                    self.current_imsize,
                    **self.cfg.models.generator.unet))
        if self._one_hot_pose:
            self.decoder.add_module(
                f"pose_concat{i}", layers.OneHotPoseConcat())
        self.decoder.add_module(
            f"basic_block{i}", self.create_up_block(self.current_imsize))
        self.decoder.add_module(
            "to_rgb", to_rgb
        )
        if self.progressive_enabled:
            self.decoder.add_module(
                "transition_block", layers.TransitionBlock())

    def extend_encoder(self):
        from_rgb, *old_blocks = self.encoder
        from_rgb.extend()
        encoder = nn.ModuleList([])
        encoder.add_module("from_rgb", from_rgb)
        i = self.transition_step
        # New block
        encoder.add_module(
            f"basic_block{i}", self.create_down_block(self.current_imsize))
        encoder.add_module(f"downsample{i}", blocks.build_avgpool(
            self.cfg.models.generator.conv2d_config, kernel_size=2))
        if self.progressive_enabled:
            encoder.add_module("transition_block", layers.TransitionBlock())

        for name, module in self.encoder.named_children():
            if isinstance(name, layers.TransitionBlock):
                continue
            if isinstance(name, FromRGB):
                continue
            encoder.add_module(name, module)
        self.encoder = encoder

    def extend(self):
        super().extend()
        self.extend_encoder()
        self.extend_decoder()

    def create_down_block(self, resolution):
        return blocks.BasicBlock(
            self.cfg.models.generator.conv2d_config, self.current_imsize,
            self.conv_channel_size(),
            [self.conv_channel_size(), self.prev_conv_channel_size()],
            residual=self.cfg.models.generator.residual)

    def create_up_block(self, resolution):
        if self.current_imsize == self.min_fmap_resolution:
            start_size = self.conv_channel_size() + self.z_shape[0]
        else:
            start_size = self.prev_conv_channel_size()
        if self.cfg.models.generator.scalar_pose_input:
            if self.current_imsize == self.min_fmap_resolution:
                start_size += 16
        else:
            start_size += self.cfg.models.pose_size // 2
        return blocks.BasicBlock(
            self.conv2d_config, self.current_imsize,
            start_size, [start_size, self.conv_channel_size()],
            residual=self.cfg.models.generator.residual)

    def forward_encoder(self, x, mask, batch):
        if self.concat_input_mask:
            x = torch.cat((x, mask, 1 - mask), dim=1)
        unet_features = {}
        for module in self.encoder:
            x, mask, batch = module((x, mask, batch))
            if isinstance(module, blocks.BasicBlock):
                unet_features[module._resolution] = (x, mask)
        return x, mask, unet_features

    def forward_decoder(self, x, mask, batch):
        for module in self.decoder:
            x, mask, batch = module((x, mask, batch))
        return x, mask

    def forward(
            self,
            condition,
            mask, landmarks=None, z=None,
            **kwargs):
        if z is None:
            z = self.generate_latent_variable(condition)
        landmarks_oh = None
        if self._one_hot_pose:
            landmarks_oh = generate_pose_channel_images(
                4, self.current_imsize, condition.device, landmarks,
                condition.dtype)
        batch = dict(
            landmarks=landmarks,
            landmarks_oh=landmarks_oh,
            z=z,
            transition_value=self.transition_value)
        orig_mask = mask
        mask = self._get_input_mask(condition, mask)

        x, mask, unet_features = self.forward_encoder(condition, mask, batch)

        batch = dict(
            landmarks=landmarks,
            landmarks_oh=landmarks_oh,
            z=z,
            unet_features=unet_features,
            transition_value=self.transition_value)
        x, mask = self.forward_decoder(x, mask, batch)

        if self.cfg.models.generator.use_skip:
            x = condition * orig_mask + (1 - orig_mask) * x
        return x


if __name__ == "__main__":
    from deep_privacy.config import Config, default_parser
    args = default_parser().parse_args()
    cfg = Config.fromfile(args.config_path)

    g = Generator(cfg).cuda()
    g.extend()
    g.cuda()
    imsize = g.current_imsize
    batch = dict(
        mask=torch.ones((8, 1, imsize, imsize)).cuda(),
        condition=torch.randn((8, 3, imsize, imsize)).cuda(),
        landmarks=torch.randn((8, 14)).cuda()
    )
