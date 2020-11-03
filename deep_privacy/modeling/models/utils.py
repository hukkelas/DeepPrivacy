import torch

batch_indexes = {

}
pose_indexes = {

}


def transition_features(x_old, x_new, transition_variable):
    assert x_old.shape == x_new.shape, "Old shape: {}, New: {}".format(
        x_old.shape, x_new.shape)
    return torch.lerp(x_old, x_new, transition_variable)


def get_transition_value(x_old, x_new, transition_variable):
    assert x_old.shape == x_new.shape, "Old shape: {}, New: {}".format(
        x_old.shape, x_new.shape)
    return torch.lerp(x_old, x_new, transition_variable)


def generate_pose_channel_images(
        min_imsize, max_imsize, device, pose_information, dtype):
    pose_information = pose_information.clone()
    batch_size = pose_information.shape[0]
    assert pose_information.shape[1] > 0, pose_information.shape
    num_poses = pose_information.shape[1] // 2
    pose_x = pose_information[:, range(
        0, pose_information.shape[1], 2)].view(-1)
    pose_y = pose_information[:, range(
        1, pose_information.shape[1], 2)].view(-1)
    assert batch_size <= 256, \
        f"Overflow error for batch size > 256. Was: {batch_size}"
    if (max_imsize, batch_size) not in batch_indexes.keys():
        batch_indexes[(max_imsize, batch_size)] = torch.cat(
            [torch.ones(num_poses, dtype=torch.long) * k for k in range(batch_size)])
        pose_indexes[(max_imsize, batch_size)] = torch.arange(
            0, num_poses).repeat(batch_size)
    batch_idx = batch_indexes[(max_imsize, batch_size)]
    pose_idx = pose_indexes[(max_imsize, batch_size)].clone()
    # All poses that are outside image, we move to the last pose channel
    illegal_mask = ((pose_x < 0) + (pose_x >= 1.0) +
                    (pose_y < 0) + (pose_y >= 1.0)) != 0
    pose_idx[illegal_mask] = num_poses
    pose_x[illegal_mask] = 0
    pose_y[illegal_mask] = 0
    pose_images = {}
    imsize = min_imsize
    while imsize <= max_imsize:
        new_im = torch.zeros((batch_size, num_poses + 1, imsize, imsize),
                             dtype=dtype, device=device)

        px = (pose_x * imsize).long()
        py = (pose_y * imsize).long()
        new_im[batch_idx, pose_idx, py, px] = 1
        new_im = new_im[:, :-1]  # Remove "throwaway" channel
        pose_images[imsize] = new_im
        imsize *= 2
    return pose_images


class NetworkWrapper(torch.nn.Module):

    def __init__(self, network):
        super().__init__()
        self.network = network
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.forward_block = torch.nn.DataParallel(
                self.network
            )
        else:
            self.forward_block = self.network

    def forward(self, *inputs, **kwargs):
        return self.forward_block(*inputs, **kwargs)

    def extend(self):
        self.network.extend()

    def update_transition_value(self, value):
        self.network.transition_value = value

    def new_parameters(self):
        return self.network.new_parameters()

    def state_dict(self):
        return self.network.state_dict()

    def load_state_dict(self, ckpt):
        self.network.load_state_dict(ckpt)

    @property
    def current_imsize(self):
        return self.network.current_imsize

    def forward_fake(self, condition, mask, landmarks=None,
                     fake_img=None, with_pose=False, **kwargs):
        return self(
            fake_img, condition, mask, landmarks, with_pose=with_pose
        )

    def generate_latent_variable(self, *args, **kwargs):
        return self.network.generate_latent_variable(*args, **kwargs)

    def forward_train(self, *args, **kwargs):
        return [self(*args, **kwargs)]

    def update_beta(self, *args, **kwargs):
        return self.network.update_beta(*args, **kwargs)

    @property
    def ra_beta(self):
        return self.network.ra_beta

    def update_ra(self, *args, **kwargs):
        return self.network.update_ra(*args, **kwargs)

    @property
    def z_shape(self):
        return self.network.z_shape
