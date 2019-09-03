import os
import shutil
import torch
from apex.amp._amp_state import _amp_state
from deep_privacy.models.discriminator import Discriminator, DeepDiscriminator
from deep_privacy.models.generator import Generator

def save_checkpoint(state, save_path, is_best=False, max_keep=None):
    # save checkpoint
    torch.save(state, save_path)

    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, 'latest_checkpoint')

    save_path = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_path + '\n'] + ckpt_list
    else:
        ckpt_list = [save_path + '\n']

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            ckpt = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt):
                os.remove(ckpt)
        ckpt_list[max_keep:] = []

    with open(list_path, 'w') as f:
        f.writelines(ckpt_list)

    # copy best
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, 'best_model.ckpt'))

def load_checkpoint(ckpt_dir_or_file, load_best=False, map_location=None):
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, 'best_model.ckpt')
        else:
            with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint')) as f:
                if False:
                    ckpt_paths = f.readlines()
                    ckpt_path = [_.strip() for _ in ckpt_paths if "30000000" in _][0]
                    ckpt_path = os.path.join(ckpt_dir_or_file, ckpt_path)
                else:
                    ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt

def clip(tensor, min_val, max_val):
    tensor[tensor < min_val] = min_val
    tensor[tensor > max_val] = max_val
    return tensor

def flip_horizontal(images):
    # Flip on -1 dimension
    idx = torch.arange(images.shape[-1] -1, -1, -1 ,dtype=torch.long)
    return images[:, :, :, idx]

def _rampup(epoch, rampup_length):
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return np.exp(-p*p*5.0)
    else:
        return 1.0

def _rampdown_linear(epoch, num_epochs, rampdown_length):
    if epoch >= num_epochs - rampdown_length:
        return float(num_epochs - epoch) / rampdown_length
    else:
        return 1.0

def init_model(start_channel_size, num_levels, model):
    transition_channels = [
        start_channel_size,
        start_channel_size,
        start_channel_size,
        start_channel_size//2,
        start_channel_size//4,
        start_channel_size//8,
        start_channel_size//16,
        start_channel_size//32,
    ]
    for i in range(num_levels):
        model.extend(transition_channels[i])


def truncated_normal(mean, std, size):
    normal = torch.zeros(size).normal_(mean=mean, std=std)
    return normal

def amp_state_has_overflow():
    for loss_scaler in _amp_state.loss_scalers:
        if loss_scaler._has_overflow:
            return True
    return False

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

    def forward(self, *inputs):
        return self.forward_block(*inputs)

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


def wrap_models(models):
    if isinstance(models, tuple) or isinstance(models, list):
        return [NetworkWrapper(x) for x in models]
    return NetworkWrapper(models)

def compute_transition_value(global_step, is_transitioning, transition_iters, latest_switch):
    transition_variable = 1
    if is_transitioning:
        diff = global_step - latest_switch
        transition_variable = diff / transition_iters
    return transition_variable
