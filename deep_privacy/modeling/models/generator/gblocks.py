import torch
import numpy as np
import torch.nn as nn
from .. import blocks


class LatentVariableConcat(nn.Module):

    def __init__(self, conv2d_config):
        super().__init__()

    def forward(self, _inp):
        x, mask, batch = _inp
        z = batch["z"]
        x = torch.cat((x, z), dim=1)
        return (x, mask, batch)


class UnetSkipConnection(nn.Module):

    def __init__(self, conv2d_config: dict, in_channels: int,
                 out_channels: int, resolution: int,
                 residual: bool, enabled: bool):
        super().__init__()
        self.use_iconv = conv2d_config.conv.type == "iconv"
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._resolution = resolution
        self._enabled = enabled
        self._residual = residual
        if self.use_iconv:
            self.beta0 = torch.nn.Parameter(torch.tensor(1.))
            self.beta1 = torch.nn.Parameter(torch.tensor(1.))
        else:
            if self._residual:
                self.conv = blocks.build_base_conv(
                    conv2d_config, False, in_channels // 2,
                    out_channels, kernel_size=1, padding=0)
            else:
                self.conv = blocks.ConvAct(
                    conv2d_config, in_channels, out_channels,
                    kernel_size=1, padding=0)

    def forward(self, _inp):
        if not self._enabled:
            return _inp
        x, mask, batch = _inp
        skip_x, skip_mask = batch["unet_features"][self._resolution]
        assert x.shape == skip_x.shape, (x.shape, skip_x.shape)
        del batch["unet_features"][self._resolution]
        if self.use_iconv:
            denom = skip_mask * self.beta0.relu() + mask * self.beta1.relu() + 1e-8
            gamma = skip_mask * self.beta0.relu() / denom
            x = skip_x * gamma + (1 - gamma) * x
            mask = skip_mask * gamma + (1 - gamma) * mask
        else:
            if self._residual:
                skip_x, skip_mask = self.conv((skip_x, skip_mask))
                x = (x + skip_x) / np.sqrt(2)
                if self._probabilistic:
                    mask = (mask + skip_mask) / np.sqrt(2)
            else:
                x = torch.cat((x, skip_x), dim=1)
                x, mask = self.conv((x, mask))
        return x, mask, batch

    def __repr__(self):
        return " ".join([
            self.__class__.__name__,
            f"In channels={self._in_channels}",
            f"Out channels={self._out_channels}",
            f"Residual: {self._residual}",
            f"Enabled: {self._enabled}"
            f"IConv: {self.use_iconv}"
        ])
