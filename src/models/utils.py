import torch
import torch.nn as nn

batch_indexes = {

}
pose_indexes = {

}

def generate_pose_channel_images(min_imsize, max_imsize, device, pose_information, dtype):
    batch_size = pose_information.shape[0]
    if pose_information.shape[1] == 2:
        pose_images = []
        imsize = min_imsize
        while imsize <= max_imsize:
            new_im = torch.zeros((batch_size, 1, imsize, imsize), dtype=dtype, device=device)
            imsize *= 2
            pose_images.append(new_im)
        return pose_images
    num_poses = pose_information.shape[1] // 2
    pose_x = pose_information[:, range(0, pose_information.shape[1], 2)].view(-1)
    pose_y = pose_information[:, range(1, pose_information.shape[1], 2)].view(-1)
    assert batch_size <= 256, "Overflow error for batch size > 256"
    if (max_imsize, batch_size) not in batch_indexes.keys():
        batch_indexes[(max_imsize, batch_size)] = torch.cat(
            [torch.ones(num_poses, dtype=torch.long)*k for k in range(batch_size)])
        pose_indexes[(max_imsize, batch_size)] = torch.arange(0, num_poses).repeat(batch_size)
    batch_idx = batch_indexes[(max_imsize, batch_size)]
    pose_idx = pose_indexes[(max_imsize, batch_size)].clone()
    # All poses that are outside image, we move to the last pose channel
    illegal_mask = ((pose_x < 0) + (pose_x >= 1.0) + (pose_y < 0) + (pose_y >= 1.0)) != 0
    pose_idx[illegal_mask] = num_poses
    pose_x[illegal_mask] = 0
    pose_y[illegal_mask] = 0
    pose_images = []
    imsize = min_imsize
    while imsize <= max_imsize:
        new_im = torch.zeros((batch_size, num_poses+1, imsize, imsize), dtype=dtype, device=device)

        px = (pose_x * imsize).long()
        py = (pose_y * imsize).long()
        new_im[batch_idx, pose_idx, py, px] = 1
        new_im = new_im[:, :-1] # Remove "throwaway" channel
        pose_images.append(new_im)
        imsize *= 2
    return pose_images

