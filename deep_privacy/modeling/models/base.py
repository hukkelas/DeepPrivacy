import torch
from torch import nn
from . import blocks, layers
from deep_privacy import torch_utils


def transition_features(x_old, x_new, transition_variable):
    assert x_old.shape == x_new.shape,\
        "Old shape: {}, New: {}".format(x_old.shape, x_new.shape)
    return torch.lerp(x_old, x_new, transition_variable)


class Module(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def extra_repr(self):
        num_params = torch_utils.number_of_parameters(self) / 10**6

        return f"Number of parameters: {num_params:.3f}M"


class ProgressiveBase(Module):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.transition_value = 1.0
        self.min_fmap_resolution = min(self.cfg.models.conv_size.keys())
        self.current_imsize = self.min_fmap_resolution
        self.transition_step = 0
        self.progressive_enabled = self.cfg.trainer.progressive.enabled
        self.conv_size = self.cfg.models.conv_size
        self.conv_size = {int(k): v for k, v in self.conv_size.items()}

    def extend(self):
        self.transition_step += 1
        self.current_imsize *= 2
        for child in self.children():
            if isinstance(child, ProgressiveBase):
                child.extend()

    def update_transition_value(self, value: float):
        self.transition_value = value

    def conv_channel_size(self):
        return self.conv_size[self.current_imsize]

    def prev_conv_channel_size(self):
        return self.conv_size[self.current_imsize // 2]

    def state_dict(self, *args, **kwargs):
        return {
            "transition_step": self.transition_step,
            "transition_value": self.transition_value,
            "parameters": super().state_dict(*args, **kwargs)
        }

    def load_state_dict(self, ckpt):
        for i in range(ckpt["transition_step"] - self.transition_step):
            self.extend()
        self.transition_value = ckpt["transition_value"]
        super().load_state_dict(ckpt["parameters"])


class FromRGB(ProgressiveBase):

    def __init__(self, cfg, conv2d_config, in_channels, current_imsize=None):
        super().__init__(cfg)
        if current_imsize is not None:
            self.current_imsize = current_imsize
        self.progressive = cfg.trainer.progressive.enabled
        self._conv2d_config = conv2d_config
        self._in_channels = in_channels
        self.conv = blocks.ConvAct(
            conv2d_config, in_channels,
            self.conv_channel_size(), kernel_size=1, padding=0
        )
        self.old_conv = nn.Sequential()

    def extend(self):
        super().extend()
        if self.progressive:
            self.old_conv = nn.Sequential(
                layers.AvgPool2d(kernel_size=2),
                self.conv
            )
        self.conv = blocks.ConvAct(
            self._conv2d_config, self._in_channels,
            self.conv_channel_size(), 1, padding=0
        )

    def forward(self, _inp):
        x_, mask_, batch = _inp
        x, mask = self.conv((x_, mask_))
        if not self.progressive:
            return (x, mask, batch)
        x_old, mask_old = self.old_conv((x_, mask_))
        batch["x_old"] = x_old
        batch["mask_old"] = mask_old
        return x, mask, batch
