import deep_privacy.config_parser as config_parser
import torch
import os
import torchvision
from deep_privacy.models.unet_model import init_model
from deep_privacy.data_tools.dataloaders import load_dataset
from deep_privacy.data_tools.data_utils import denormalize_img

dl_train, _ = load_dataset("yfcc100m128", 
                           batch_size=64, 
                           imsize=64,
                           full_validation=False, 
                           pose_size=14, 
                           load_fraction=True)
config = config_parser.load_config("models/minibatch_std/config.yml")
ckpt = torch.load("models/minibatch_std/transition_checkpoints/imsize64.ckpt")
discriminator, generator = init_model(
    config.models.pose_size,
    config.models.start_channel_size,
    config.models.image_channels,
    config.models.discriminator.structure
)
generator.load_state_dict(ckpt["G"])
generator.cuda()
print(generator.network.current_imsize)
dl_train.update_next_transition_variable(1.0)
ims, conditions, landmarks = next(iter(dl_train))

fakes = denormalize_img(generator(conditions, landmarks))
os.makedirs(".debug", exist_ok=True)
torchvision.utils.save_image(fakes, ".debug/test.jpg")

# Extend
generator.extend()
generator.cuda()
generator.transition_value = 0.0
dl_train, _ = load_dataset("yfcc100m128", 
                           batch_size=64, 
                           imsize=128,
                           full_validation=False, 
                           pose_size=14, 
                           load_fraction=True)
dl_train.update_next_transition_variable(0.0)
ims, conditions, landmarks = next(iter(dl_train))
fakes = denormalize_img(generator(conditions, landmarks))
torchvision.utils.save_image(fakes, ".debug/after.jpg")

print("Saved images!")
#print(generator.network.current_imsize)

