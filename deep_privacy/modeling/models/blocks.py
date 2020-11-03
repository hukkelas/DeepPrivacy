import torch.nn as nn
import numpy as np
import torch
from typing import List
from . import layers
from . import iconv


def get_conv(ctype, post_act):
    type2conv = {
        "conv": layers.Conv2d,
        "iconv": iconv.IConv,
        "gconv": GatedConv
    }
    # Do not apply for output layer
    if not post_act and ctype in ["gconv", "iconv"]:
        return type2conv["conv"]
    assert ctype in type2conv
    return type2conv[ctype]


def build_base_conv(
        conv2d_config, post_act: bool, *args, **kwargs) -> nn.Conv2d:
    for k, v in conv2d_config.conv.items():
        assert k not in kwargs
        kwargs[k] = v
    # Demodulation should not be used for output layers.
    demodulation = conv2d_config.normalization == "demodulation" and post_act
    kwargs["demodulation"] = demodulation
    conv = get_conv(conv2d_config.conv.type, post_act)
    return conv(*args, **kwargs)


def build_post_activation(in_channels, conv2d_config) -> List[nn.Module]:
    _layers = []
    negative_slope = conv2d_config.leaky_relu_nslope
    _layers.append(layers.LeakyReLU(negative_slope, inplace=True))
    if conv2d_config.normalization == "pixel_wise":
        _layers.append(layers.PixelwiseNormalization())
    return _layers


def build_avgpool(conv2d_config, kernel_size) -> nn.AvgPool2d:
    if conv2d_config.conv.type == "iconv":
        return iconv.DownSample()
    return layers.AvgPool2d(kernel_size)


def build_convact(conv2d_config, *args, **kwargs):
    conv = build_base_conv(conv2d_config, True, *args, **kwargs)
    out_channels = conv.out_channels
    post_act = build_post_activation(out_channels, conv2d_config)
    return nn.Sequential(conv, *post_act)


class ConvAct(nn.Module):

    def __init__(self, conv2d_config, *args, **kwargs):
        super().__init__()
        self._conv2d_config = conv2d_config
        conv = build_base_conv(conv2d_config, True, *args, **kwargs)
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        _layers = [conv]
        _layers.extend(build_post_activation(self.out_channels, conv2d_config))
        self.layers = nn.Sequential(*_layers)

    def forward(self, _inp):
        return self.layers(_inp)


class GatedConv(layers.Conv2d):

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        out_channels *= 2
        super().__init__(in_channels, out_channels, *args, **kwargs)
        assert self.out_channels % 2 == 0
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def conv2d_forward(self, x, weight, bias=True):
        x_ = super().conv2d_forward(x, weight, bias)
        x = x_[:, :self.out_channels // 2]
        y = x_[:, self.out_channels // 2:]
        x = self.lrelu(x)
        y = y.sigmoid()
        assert x.shape == y.shape, f"{x.shape}, {y.shape}"
        return x * y


class BasicBlock(nn.Module):

    def __init__(
            self, conv2d_config, resolution: int, in_channels: int,
            out_channels: List[int], residual: bool):
        super().__init__()
        assert len(out_channels) == 2
        self._resolution = resolution
        self._residual = residual
        self.out_channels = out_channels
        _layers = []
        _in_channels = in_channels
        for out_ch in out_channels:
            conv = build_base_conv(
                conv2d_config, True, _in_channels, out_ch, kernel_size=3,
                resolution=resolution)
            _layers.append(conv)
            _layers.extend(build_post_activation(_in_channels, conv2d_config))
            _in_channels = out_ch
        self.layers = nn.Sequential(*_layers)
        if self._residual:
            self.residual_conv = build_base_conv(
                conv2d_config, post_act=False, in_channels=in_channels,
                out_channels=out_channels[-1],
                kernel_size=1, padding=0)
            self.const = 1 / np.sqrt(2)

    def forward(self, _inp):
        x, mask, batch = _inp
        y = x
        mask_ = mask
        assert y.shape[-1] == self._resolution or y.shape[-1] == 1
        y, mask = self.layers((x, mask))
        if self._residual:
            residual, mask_ = self.residual_conv((x, mask_))
            y = (y + residual) * self.const
            mask = (mask + mask_) * self.const
        return y, mask, batch

    def extra_repr(self):
        return f"Residual={self._residual}, Resolution={self._resolution}"


class PoseNormalize(nn.Module):

    @torch.no_grad()
    def forward(self, x):
        return x * 2 - 1


class ScalarPoseFCNN(nn.Module):

    def __init__(self, pose_size, hidden_size,
                 output_shape):
        super().__init__()
        pose_size = pose_size
        self._hidden_size = hidden_size
        output_size = np.prod(output_shape)
        self.output_shape = output_shape
        self.pose_preprocessor = nn.Sequential(
            PoseNormalize(),
            layers.Linear(pose_size, hidden_size),
            nn.LeakyReLU(.2),
            layers.Linear(hidden_size, output_size),
            nn.LeakyReLU(.2)
        )

    def forward(self, _inp):
        x, mask, batch = _inp
        pose_info = batch["landmarks"]
        del batch["landmarks"]
        pose = self.pose_preprocessor(pose_info)
        pose = pose.view(-1, *self.output_shape)
        if x.shape[0] == 1 and x.shape[2] == 1 and x.shape[3] == 1:
            # Analytical normalization propagation
            pose = pose.mean(dim=2, keepdim=True).mean(dim=3, keepdims=True)
        x = torch.cat((x, pose), dim=1)
        return x, mask, batch

    def __repr__(self):
        return " ".join([
            self.__class__.__name__,
            f"hidden_size={self._hidden_size}",
            f"output shape={self.output_shape}"
        ])


class Attention(nn.Module):

    def __init__(self, in_channels):
        super(Attention, self).__init__()
        # Channel multiplier
        self.in_channels = in_channels
        self.theta = layers.Conv2d(
            self.in_channels, self.in_channels // 8, kernel_size=1, padding=0,
            bias=False)
        self.phi = layers.Conv2d(
            self.in_channels, self.in_channels // 8, kernel_size=1, padding=0,
            bias=False)
        self.g = layers.Conv2d(
            self.in_channels, self.in_channels // 2, kernel_size=1, padding=0,
            bias=False)
        self.o = layers.Conv2d(
            self.in_channels // 2, self.in_channels, kernel_size=1, padding=0,
            bias=False)
        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, _inp):
        x, mask, batch = _inp
        # Apply convs
        theta, _ = self.theta((x, None))
        phi = nn.functional.max_pool2d(self.phi((x, None))[0], [2, 2])
        g = nn.functional.max_pool2d(self.g((x, None))[0], [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self.in_channels // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.in_channels // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.in_channels // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = nn.functional.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path

        o = self.o((torch.bmm(g, beta.transpose(1, 2)).view(-1,
                                                            self.in_channels // 2, x.shape[2], x.shape[3]), None))[0]
        return self.gamma * o + x, mask, batch
