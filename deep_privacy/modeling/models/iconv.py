import torch
from . import layers

eps = 1e-2


pconv_settings = {
    4: {"dilation": 1, "kernel_size": 3},
    8: {"dilation": 1, "kernel_size": 3},
    16: {"dilation": 1, "kernel_size": 5},
    32: {"dilation": 1, "kernel_size": 5},
    64: {"dilation": 1, "kernel_size": 5},
    128: {"dilation": 1, "kernel_size": 5},
    256: {"dilation": 1, "kernel_size": 5},
    None: {"dilation": 1, "kernel_size": 5}
}


class DownSample(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(2)

    def forward(self, _inp):
        x, mask, *args = _inp
        new_mask = self.pool(mask)
        x = x * mask
        x = self.pool(x) / (new_mask + eps)
        if len(args) > 0:
            return (x, new_mask, *args)
        return x, new_mask


class ExpectedValue(torch.nn.Module):

    def __init__(self, num_channels: int, resolution: int):
        super().__init__()
        stride = 1
        kernel_size = pconv_settings[resolution]["kernel_size"]
        dilation = pconv_settings[resolution]["dilation"]
        padding = layers.get_padding(kernel_size, dilation, stride)
        self._resolution = resolution
        if dilation > 1:
            self.avg_pool = torch.nn.Conv2d(
                1, 1, kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                dilation=dilation
            )
            self.avg_pool.weight.requires_grad = False
            self.avg_pool.weight.data = self.avg_pool.weight.data.zero_() + 1 / kernel_size**2
        else:
            self.avg_pool = torch.nn.AvgPool2d(
                kernel_size, stride, padding=padding)
        self.input_updater = layers.Conv2d(
            num_channels, num_channels, kernel_size, stride, padding,
            groups=num_channels, dilation=dilation, wsconv=True
        )

    def extra_repr(self, *args, **kwargs):
        return f"resolution: {self._resolution}"

    def forward(self, x, mask):
        weighted = x * mask
        weighted, _ = self.input_updater((weighted, None))
        prob_sum = self.avg_pool(mask) + eps
        return weighted / prob_sum


class IConv(layers.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, dilation=1, resolution=None, *args, **kwargs):
        super().__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, *args, **kwargs)
        self.return_mask = True
        self.conv1x1 = kernel_size == 1 and padding in [0, None]

        self.expected_value_updater = ExpectedValue(
            in_channels, resolution=resolution)
        if self.conv1x1:
            return
        self.kernel_size = kernel_size

        self.mask_updater = torch.nn.Conv2d(
            1, 1,
            self.kernel_size,
            padding=layers.get_padding(kernel_size, dilation, 1),
            dilation=dilation, bias=False)
        self.mask_updater.weight.data = self.mask_updater.weight.data.zero_() + 1
        self.mask_activation = torch.nn.Sigmoid()

    def forward(self, _inp):
        x, mask_in = _inp
        expected_x = self.expected_value_updater(x, mask_in)
        predicted_x = x * mask_in + expected_x * (1 - mask_in)
        output, mask_in = super().forward((predicted_x, mask_in))
        if self.conv1x1:
            return output, mask_in

        new_mask = self.mask_updater(mask_in)
        new_mask = self.mask_activation(new_mask)
        new_mask = (new_mask - 0.5) * 2 + 1e-6
        new_mask = new_mask.clamp(0, 1)  # Fix potential roundoff errors.
        assert output.shape[2:] == new_mask.shape[2:],\
            f"Output shape: {output.shape}, new_mask: {new_mask.shape}"
        return output, new_mask
