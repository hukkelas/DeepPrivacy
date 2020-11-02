import torch
import torch.nn as nn
import numpy as np


def get_padding(kernel_size: int, dilation: int, stride: int):
    out = (dilation * (kernel_size - 1) - 1) / 2 + 1
    return int(np.floor(out))


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 demodulation=False, wsconv=False, gain=1,
                 *args, **kwargs):
        if padding is None:
            padding = get_padding(kernel_size, dilation, stride)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        self.demodulation = demodulation
        self.wsconv = wsconv
        if self.wsconv:
            fan_in = np.prod(self.weight.shape[1:]) / self.groups
            self.ws_scale = gain / np.sqrt(fan_in)
            nn.init.normal_(self.weight)
        if bias:
            nn.init.constant_(self.bias, val=0)
        assert not self.padding_mode == "circular",\
            "conv2d_forward does not support circular padding. Look at original pytorch code"

    def _get_weight(self):
        weight = self.weight
        if self.wsconv:
            weight = self.ws_scale * weight
        if self.demodulation:
            demod = torch.rsqrt(weight.pow(2).sum([1, 2, 3]) + 1e-7)
            weight = weight * demod.view(self.out_channels, 1, 1, 1)
        return weight

    def conv2d_forward(self, x, weight, bias=True):
        bias_ = None
        if bias:
            bias_ = self.bias
        return nn.functional.conv2d(x, weight, bias_, self.stride,
                                    self.padding, self.dilation, self.groups)

    def forward(self, _inp):
        x, mask = _inp
        weight = self._get_weight()
        return self.conv2d_forward(x, weight), mask

    def __repr__(self):
        return ", ".join([
            super().__repr__(),
            f"Demodulation={self.demodulation}",
            f"Weight Scale={self.wsconv}",
            f"Bias={self.bias is not None}"
        ])


class LeakyReLU(nn.LeakyReLU):

    def forward(self, _inp):
        x, mask = _inp
        return super().forward(x), mask


class AvgPool2d(nn.AvgPool2d):

    def forward(self, _inp):
        x, mask, *args = _inp
        x = super().forward(x)
        mask = super().forward(mask)
        if len(args) > 0:
            return (x, mask, *args)
        return x, mask


def up(x):
    if x.shape[0] == 1 and x.shape[2] == 1 and x.shape[3] == 1:
        # Analytical normalization
        return x
    return nn.functional.interpolate(
        x, scale_factor=2, mode="nearest")


class NearestUpsample(nn.Module):

    def forward(self, _inp):
        x, mask, *args = _inp
        x = up(x)
        mask = up(mask)
        if len(args) > 0:
            return (x, mask, *args)
        return x, mask


class PixelwiseNormalization(nn.Module):

    def forward(self, _inp):
        x, mask = _inp
        norm = torch.rsqrt((x**2).mean(dim=1, keepdim=True) + 1e-7)
        return x * norm, mask


class Linear(nn.Linear):

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.linear = nn.Linear(in_features, out_features)
        fanIn = in_features
        self.wtScale = 1 / np.sqrt(fanIn)

        nn.init.normal_(self.weight)
        nn.init.constant_(self.bias, val=0)

    def _get_weight(self):
        return self.weight * self.wtScale

    def forward_linear(self, x, weight):
        return nn.functional.linear(x, weight, self.bias)

    def forward(self, x):
        return self.forward_linear(x, self._get_weight())


class OneHotPoseConcat(nn.Module):

    def forward(self, _inp):
        x, mask, batch = _inp
        landmarks = batch["landmarks_oh"]
        res = x.shape[-1]
        landmark = landmarks[res]
        x = torch.cat((x, landmark), dim=1)
        del batch["landmarks_oh"][res]
        return x, mask, batch


def transition_features(x_old, x_new, transition_variable):
    assert x_old.shape == x_new.shape,\
        "Old shape: {}, New: {}".format(x_old.shape, x_new.shape)
    return torch.lerp(x_old.float(), x_new.float(), transition_variable)


class TransitionBlock(nn.Module):

    def forward(self, _inp):
        x, mask, batch = _inp
        x = transition_features(
            batch["x_old"], x, batch["transition_value"])
        mask = transition_features(
            batch["mask_old"], mask, batch["transition_value"])
        del batch["x_old"]
        del batch["mask_old"]
        return x, mask, batch
