import sys
from unet_model import Generator
from train import NetworkWrapper
from utils import to_cuda, init_model
import numpy as np
import matplotlib.pyplot as plt
from options import print_options
import cv2
def draw_bboxes(image, bboxes, colors):
    image = image.copy()
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        image = cv2.rectangle(image, (x0, y0), (x1, y1),colors, 1)
    return image


def draw_keypoints(image, keypoints, colors):
    image = image.copy()
    for keypoint in keypoints:
        X = keypoint[0, :3]
        Y = keypoint[1, :3]
        for x, y in zip(X, Y):
            cv2.circle(image, (x, y), 2, colors)
    return image


def image_to_numpy(images):
    single_image = False
    if len(images.shape) == 3:
        single_image = True
        images = images[None]
    images = images.data.detach().cpu().numpy()
    r,g,b = images[:, 0], images[:, 1], images[:, 2]
    images = np.stack((r,g,b), axis=3)
    if single_image:
        return images[0]
    return images

def init_generator(checkpoint):
    start_channel_size = checkpoint["start_channel_size"]
    print("Start channel size:", start_channel_size)
    pose_dim = checkpoint["pose_size"]
    image_channels = checkpoint["image_channels"]
    transition_step = checkpoint["transition_step"]
    transition_value = checkpoint["transition_variable"]
    g = Generator(pose_dim, start_channel_size, image_channels)
    init_model(start_channel_size, transition_step, g)
    g = to_cuda(g)
    g.load_state_dict(checkpoint["running_average_generator"])
    g.transition_value = transition_value
    print("Transition step:", transition_step)
    print("Transition value:", transition_value)
    print("Global step:", checkpoint["global_step"])
    print_options(checkpoint)
    return g


def get_model_name():
    argv = sys.argv
    assert len(sys.argv) == 2, "Expected argument length of 1. Run the script with \"python -m scripts.generate_final_images\" [model_name]"
    return argv[-1]


def plot_pose(pose, imsize, x_offset=0):
    pose = pose.squeeze() * imsize
    assert pose.size == pose.shape[0]
    x = pose[range(0, len(pose,), 2)]
    y = pose[range(1, len(pose), 2)]
    plt.plot(x + x_offset, y, "o")
