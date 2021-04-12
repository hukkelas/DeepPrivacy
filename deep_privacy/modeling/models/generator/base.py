import torch
from ..base import Module


class RunningAverageGenerator(Module):

    def __init__(self, z_shape, conv2d_config: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.z_shape = z_shape
        self.conv2d_config = conv2d_config

    def update_beta(self, batch_size: int):
        self.ra_beta = 0.5 ** (batch_size / (10 * 1000))

    @torch.no_grad()
    def update_ra(self, normal_generator):
        """
            Update running average generator
        """
        for avg_param, cur_param in zip(self.parameters(),
                                        normal_generator.parameters()):
            assert avg_param.shape == cur_param.shape
            avg_param.data = self.ra_beta * avg_param + \
                (1 - self.ra_beta) * cur_param

    def forward_train(self, *args, **kwargs):
        return [self(*args, **kwargs)]

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

    def _get_input_mask(self, condition, mask):
        return mask
