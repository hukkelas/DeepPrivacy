import src.config_parser
import torch
import torchvision
from src.models.unet_model import init_model
from src.data_tools.dataloaders_v2 import load_dataset
from src.data_tools.data_utils import denormalize_img

dl_train, _ = load_dataset("yfcc100m128", 64, 8,False, 14, True)
config = config_parser.load_config("../models/test/config.yml")
ckpt = torch.load("tests/step_600064.ckpt")
discriminator, generator = init_model(
    config.models.pose_size,
    config.models.start_channel_size,
    config.models.image_channels,
    config.models.discriminator.structure
)
generator.load_state_dict(ckpt["G"])
generator.cuda()
dl_train.update_next_transition_variable(1.0)
ims, conditions, landmarks = next(iter(dl_train))

fakes = denormalize_img(generator(conditions, landmarks))
torchvision.utils.save_image(fakes, "tests/test.jpg")

# Extend
generator.extend()
generator.cuda()
generator.transition_value = 0.0
dl_train, _ = load_dataset("yfcc100m128", 64, 16,False, 14, True)
dl_train.update_next_transition_variable(0.0)
ims, conditions, landmarks = next(iter(dl_train))
fakes = denormalize_img(generator(conditions, landmarks))
torchvision.utils.save_image(fakes, "tests/after.jpg")

print("Saved images!")
#print(generator.network.current_imsize)

