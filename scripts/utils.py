import sys
from unet_model import Generator
from train import DataParallellWrapper
from utils import to_cuda, init_model
import numpy as np
def image_to_numpy(images):
    single_image = False
    if len(images.shape) == 3:
        single_image = True
        images = images[None]
    images = images.data.cpu().numpy()
    r,g,b = images[:, 0], images[:, 1], images[:, 2]
    images = np.stack((r,g,b), axis=3)
    if single_image:
        return images[0]
    return images

def init_generator(checkpoint):
    start_channel_size = checkpoint["start_channel_size"]
    print("Start channel size:", start_channel_size)
    pose_dim = checkpoint["pose_dim"]
    image_channels = checkpoint["image_channels"]
    transition_step = checkpoint["transition_step"]
    transition_value = checkpoint["transition_variable"]
    g = Generator(pose_dim, start_channel_size, image_channels)
    g = DataParallellWrapper(g)
    init_model(start_channel_size, transition_step, g)
    g = to_cuda(g)
    g.load_state_dict(checkpoint["running_average_generator"])
    g.update_transition_value(transition_value)
    print("Transition step:", transition_step)
    print("Transition value:", transition_value)
    print("Global step:", checkpoint["global_step"])
    return g


def get_model_name():
    argv = sys.argv
    assert len(sys.argv) == 2, "Expected argument length of 1. Run the script with \"python -m scripts.generate_final_images\" [model_name]"
    return argv[-1]