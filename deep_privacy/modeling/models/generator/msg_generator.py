import torch
import torch.nn as nn
from .. import layers, blocks
from ..build import GENERATOR_REGISTRY
from .base import RunningAverageGenerator
from .gblocks import LatentVariableConcat, UnetSkipConnection


@GENERATOR_REGISTRY.register_module
class MSGGenerator(RunningAverageGenerator):

    def __init__(
            self, max_imsize: int, conv_size: dict,
            image_channels: int,
            min_fmap_resolution: int,
            residual: bool,
            pose_size: int,
            unet: dict,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.removable_hooks = []
        self.rgb_convolutions = nn.ModuleDict()
        self.max_imsize = max_imsize
        self._image_channels = image_channels
        self._min_fmap_resolution = min_fmap_resolution
        self._residual = residual
        self._pose_size = pose_size
        self.current_imsize = max_imsize
        self._unet_cfg = unet
        self.concat_input_mask = self.conv2d_config.conv.type in ["conv", "gconv"]
        self.res2channels = {int(k): v for k, v in conv_size.items()}
        self._init_decoder()
        self._init_encoder()

    def _init_encoder(self):
        self.encoder = nn.ModuleList()
        imsize = self.max_imsize
        self.from_rgb = blocks.build_convact(
            self.conv2d_config,
            in_channels=self._image_channels + self.concat_input_mask*2,
            out_channels=self.res2channels[imsize],
            kernel_size=1)
        while imsize >= self._min_fmap_resolution:
            current_size = self.res2channels[imsize]
            next_size = self.res2channels[max(imsize//2, self._min_fmap_resolution)]
            block = blocks.BasicBlock(
                self.conv2d_config, imsize, current_size,
                [current_size, next_size], self._residual)
            self.encoder.add_module(f"basic_block{imsize}", block)
            if imsize != self._min_fmap_resolution:
                self.encoder.add_module(
                    f"downsample{imsize}", layers.AvgPool2d(2))
            imsize //= 2

    def _init_decoder(self):
        self.decoder = nn.ModuleList()
        self.decoder.add_module(
            "latent_concat", LatentVariableConcat(self.conv2d_config))
        if self._pose_size > 0:
            m = self._min_fmap_resolution
            pose_shape = (16, m, m)
            pose_fcnn = blocks.ScalarPoseFCNN(self._pose_size, 128, pose_shape)
            self.decoder.add_module("pose_fcnn", pose_fcnn)
        imsize = self._min_fmap_resolution
        self.rgb_convolutions = nn.ModuleDict()
        while imsize <= self.max_imsize:
            current_size = self.res2channels[max(imsize//2, self._min_fmap_resolution)]
            start_size = current_size
            if imsize == self._min_fmap_resolution:
                start_size += self.z_shape[0]
                if self._pose_size > 0:
                    start_size += 16
            else:
                self.decoder.add_module(f"upsample{imsize}", layers.NearestUpsample())
                skip = UnetSkipConnection(
                    self.conv2d_config, current_size*2, current_size, imsize,
                    **self._unet_cfg)
                self.decoder.add_module(f"skip_connection{imsize}", skip)
            next_size = self.res2channels[imsize]
            block = blocks.BasicBlock(
                self.conv2d_config, imsize, start_size, [start_size, next_size],
                residual=self._residual)
            self.decoder.add_module(f"basic_block{imsize}", block)

            to_rgb = blocks.build_base_conv(
                self.conv2d_config, False, in_channels=next_size,
                out_channels=self._image_channels, kernel_size=1)
            self.rgb_convolutions[str(imsize)] = to_rgb
            imsize *= 2
        self.norm_constant = len(self.rgb_convolutions)

    def forward_decoder(self, x, mask, batch):
        imsize_start = max(x.shape[-1] // 2, 1)
        rgb = torch.zeros(
            (x.shape[0], self._image_channels,
             imsize_start, imsize_start),
            dtype=x.dtype, device=x.device)
        mask_size = 1
        mask_out = torch.zeros(
            (x.shape[0], mask_size,
             imsize_start, imsize_start),
            dtype=x.dtype, device=x.device)
        imsize = self._min_fmap_resolution // 2
        for module in self.decoder:
            x, mask, batch = module((x, mask, batch))
            if isinstance(module, blocks.BasicBlock):
                imsize *= 2
                rgb = layers.up(rgb)
                mask_out = layers.up(mask_out)
                conv = self.rgb_convolutions[str(imsize)]
                rgb_, mask_ = conv((x, mask))
                assert rgb_.shape == rgb.shape,\
                    f"rgb_ {rgb_.shape}, rgb: {rgb.shape}"
                rgb = rgb + rgb_
        return rgb / self.norm_constant, mask_out

    def forward_encoder(self, x, mask, batch):
        if self.concat_input_mask:
            x = torch.cat((x, mask, 1 - mask), dim=1)
        unet_features = {}
        x, mask = self.from_rgb((x, mask))
        for module in self.encoder:
            x, mask, batch = module((x, mask, batch))
            if isinstance(module, blocks.BasicBlock):
                unet_features[module._resolution] = (x, mask)
        return x, mask, unet_features

    def forward(
            self,
            condition,
            mask, landmarks=None, z=None,
            **kwargs):
        if z is None:
            z = self.generate_latent_variable(condition)
        batch = dict(
            landmarks=landmarks,
            z=z)
        orig_mask = mask
        mask = self._get_input_mask(condition, mask)
        x, mask, unet_features = self.forward_encoder(condition, mask, batch)
        batch = dict(
            landmarks=landmarks,
            z=z,
            unet_features=unet_features)
        x, mask = self.forward_decoder(x, mask, batch)
        x = condition * orig_mask + (1 - orig_mask) * x
        return x

    def load_state_dict(self, state_dict, strict=True):
        if "parameters" in state_dict:
            state_dict = state_dict["parameters"]
        old_checkpoint = any("basic_block0" in key for key in state_dict)
        if not old_checkpoint:
            return super().load_state_dict(state_dict, strict=strict)
        mapping = {}
        imsize = self._min_fmap_resolution
        i = 0
        while imsize <= self.max_imsize:
            old_key = f"decoder.basic_block{i}."
            new_key = f"decoder.basic_block{imsize}."
            mapping[old_key] = new_key
            if i >= 1:
                old_key = old_key.replace("basic_block", "skip_connection")
                new_key = new_key.replace("basic_block", "skip_connection")
                mapping[old_key] = new_key
            mapping[old_key] = new_key
            old_key = f"encoder.basic_block{i}."
            new_key = f"encoder.basic_block{imsize}."
            mapping[old_key] = new_key
            old_key = "from_rgb.conv.layers.0."
            new_key = "from_rgb.0."
            mapping[old_key] = new_key
            i += 1
            imsize *= 2
        new_sd = {}
        for key, value in state_dict.items():
            old_key = key
            if "from_rgb" in key:
                new_sd[key.replace("encoder.", "").replace(".conv.layers", "")] = value
                continue
            for subkey, new_subkey in mapping.items():
                if subkey in key:
                    old_key = key
                    key = key.replace(subkey, new_subkey)

                    break
            if "decoder.to_rgb" in key:
                continue

            new_sd[key] = value
        return super().load_state_dict(new_sd, strict=strict)
        

if __name__ == "__main__":
    from deep_privacy.config import Config, default_parser
    args = default_parser().parse_args()
    cfg = Config.fromfile(args.config_path)

    g = MSGGenerator(cfg).cuda()
    g.extend()
    g.cuda()
    imsize = g.current_imsize
    batch = dict(
        mask=torch.ones((8, 1, imsize, imsize)).cuda(),
        condition=torch.randn((8, 3, imsize, imsize)).cuda(),
        landmarks=torch.randn((8, 14)).cuda()
    )
    print(g(**batch).shape)
