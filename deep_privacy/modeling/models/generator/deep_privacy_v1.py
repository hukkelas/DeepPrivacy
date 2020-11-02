import torch.nn as nn
import torch
from .. import layers, blocks
from ..utils import generate_pose_channel_images
from ..build import GENERATOR_REGISTRY
from .progressive_generator import DecoderUpsample
from .gblocks import LatentVariableConcat


class ConvAct(nn.Module):

    def __init__(self, conv2d_config, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            blocks.build_base_conv(
                conv2d_config,
                True,
                in_channels,
                out_channels,
                kernel_size),
            *blocks.build_post_activation(
                out_channels,
                conv2d_config))

    def forward(self, _inp):
        x, mask, batch = _inp
        x, mask = self.conv((x, mask))
        return (x, mask, batch)


class UnetSkipConnection(nn.Module):

    def __init__(self, cfg, in_channels: int,
                 out_channels: int, resolution: int):
        super().__init__()
        self.cfg = cfg
        conv2d_config = self.cfg.models.generator.conv2d_config
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._resolution = resolution

        self.conv = blocks.ConvAct(
            conv2d_config, in_channels, out_channels,
            kernel_size=1, padding=0)

    def forward(self, _inp):
        x, mask, batch = _inp
        skip_x, skip_mask = batch["unet_features"][self._resolution]

        del batch["unet_features"][self._resolution]
        landmarks = batch["landmarks_oh"]
        res = x.shape[-1]
        landmark = landmarks[res]
        x = torch.cat((x, skip_x, landmark), dim=1)
        del batch["landmarks_oh"][res]
        x, mask = self.conv((x, mask))
        return x, mask, batch


@GENERATOR_REGISTRY.register_module
class DeepPrivacyV1(nn.Module):

    def __init__(self, cfg, conv2d_config: dict, *args, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.current_imsize = 128
        self.init_decoder(conv2d_config)
        self.init_encoder(conv2d_config)
        self.z_shape = cfg.models.generator.z_shape

    def init_encoder(self, conv2d_config):
        encoder = nn.ModuleList()
        from_rgb = ConvAct(conv2d_config, 3, 128, 1)
        encoder.add_module("from_rgb", from_rgb)
        imsize = 128
        block5 = blocks.BasicBlock(
            conv2d_config, imsize, 128, [
                256, 256], residual=False)
        block4 = blocks.BasicBlock(
            conv2d_config, imsize // 2, 256, [512, 512],
            residual=False)
        block3 = blocks.BasicBlock(
            conv2d_config, imsize // 4, 512, [512, 512],
            residual=False)
        block2 = blocks.BasicBlock(
            conv2d_config, imsize // 8, 512, [512, 512],
            residual=False)
        block1 = blocks.BasicBlock(
            conv2d_config, imsize // 16, 512, [512, 512],
            residual=False)
        block0 = blocks.BasicBlock(
            conv2d_config, imsize // 32, 512, [512, 512],
            residual=False)
        encoder.add_module("basic_block5", block5)
        encoder.add_module("downsample5", layers.AvgPool2d(2))
        encoder.add_module("basic_block4", block4)
        encoder.add_module("downsample4", layers.AvgPool2d(2))
        encoder.add_module("basic_block3", block3)
        encoder.add_module("downsample3", layers.AvgPool2d(2))
        encoder.add_module("basic_block2", block2)
        encoder.add_module("downsample2", layers.AvgPool2d(2))
        encoder.add_module("basic_block1", block1)
        encoder.add_module("downsample1", layers.AvgPool2d(2))
        encoder.add_module("basic_block0", block0)
        self.encoder = encoder

    def init_decoder(self, conv2d_config):
        decoder = nn.ModuleList()
        imsize = 128
        decoder.add_module("pose_concat0", layers.OneHotPoseConcat())
        decoder.add_module(
            "latent_concate",
            LatentVariableConcat(conv2d_config))
        conv = ConvAct(conv2d_config, 512 + 7 + 32, 512, 1)
        decoder.add_module("conv1x1", conv)

        block0 = blocks.BasicBlock(
            conv2d_config, imsize // 32, 512, [512, 512], False)
        block1 = blocks.BasicBlock(
            conv2d_config, imsize // 16, 512, [512, 512], False)
        block2 = blocks.BasicBlock(
            conv2d_config, imsize // 8, 512, [512, 512], False)
        block3 = blocks.BasicBlock(
            conv2d_config, imsize // 4, 512, [512, 512], False)
        block4 = blocks.BasicBlock(
            conv2d_config, imsize // 2, 512, [256, 256], False)
        block5 = blocks.BasicBlock(
            conv2d_config, imsize, 256, [
                128, 128], False)
        decoder.add_module("basic_block0", block0)
        decoder.add_module("upsample0", DecoderUpsample())
        decoder.add_module(
            "skip_connection1",
            UnetSkipConnection(
                self.cfg,
                512 * 2 + 7,
                512,
                imsize // 16))

        decoder.add_module("basic_block1", block1)
        decoder.add_module("upsample1", DecoderUpsample())
        decoder.add_module(
            "skip_connection2",
            UnetSkipConnection(
                self.cfg,
                512 * 2 + 7,
                512,
                imsize // 8))

        decoder.add_module("basic_block2", block2)
        decoder.add_module("upsample2", DecoderUpsample())
        decoder.add_module(
            "skip_connection3",
            UnetSkipConnection(
                self.cfg,
                512 * 2 + 7,
                512,
                imsize // 4))

        decoder.add_module("basic_block3", block3)
        decoder.add_module("upsample3", DecoderUpsample())
        decoder.add_module(
            "skip_connection4",
            UnetSkipConnection(
                self.cfg,
                512 * 2 + 7,
                512,
                imsize // 2))

        decoder.add_module("basic_block4", block4)
        decoder.add_module("upsample4", DecoderUpsample())
        decoder.add_module(
            "skip_connection5",
            UnetSkipConnection(
                self.cfg,
                256 * 2 + 7,
                256,
                imsize))

        decoder.add_module("basic_block5", block5)
        self.to_rgb = blocks.build_base_conv(conv2d_config, False, 128, 3, 1)

        self.decoder = decoder

    def generate_latent_variable(self, *args):
        if len(args) == 1:
            x_in = args[0]

            return torch.randn(x_in.shape[0], *self.z_shape,
                               device=x_in.device,
                               dtype=x_in.dtype)
        elif len(args) == 3:
            batch_size, device, dtype = args
            return torch.randn(batch_size, *self.z_shape,
                               device=device,
                               dtype=dtype)
        raise ValueError(
            f"Expected either x_in or (batch_size, device, dtype. Got: {args}")

    def forward_encoder(self, x, mask, batch):
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
            mask, landmarks, z=None, **kwargs):
        if z is None:
            z = self.generate_latent_variable(condition)
        landmarks_oh = None
        landmarks_oh = generate_pose_channel_images(
            4, self.current_imsize, condition.device, landmarks,
            condition.dtype)
        batch = dict(
            landmarks=landmarks,
            landmarks_oh=landmarks_oh,
            z=z)
        x, mask, unet_features = self.forward_encoder(condition, mask, batch)
        batch = dict(
            landmarks=landmarks,
            landmarks_oh=landmarks_oh,
            z=z,
            unet_features=unet_features)
        x, mask = self.forward_decoder(x, mask, batch)
        x, mask = self.to_rgb((x, mask))
#        x = condition * orig_mask + (1 - orig_mask) * x
        return x
