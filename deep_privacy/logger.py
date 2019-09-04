import torch
import torchvision
import os
if torch.__version__ == "0.5.0a0":
    from tensorboardX import SummaryWriter
else:
    from torch.utils.tensorboard import SummaryWriter


PRINT_LOG_LEVEL = 1
SPAM = 0
INFO = 1
WARNING = 2


class Logger:

    def __init__(self, logdir, generated_data_dir):
        self.writer = SummaryWriter(
            os.path.join(logdir, "train")
        )
        self.validation_writer = SummaryWriter(
            os.path.join(logdir, "val"))
        self.global_step = 0
        self.image_dir = generated_data_dir

        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(os.path.join(self.image_dir, "validation"), exist_ok=True)
        os.makedirs(os.path.join(self.image_dir, "transition"), exist_ok=True)

    def update_global_step(self, global_step):
        self.global_step = global_step

    def log_variable(self, tag, value, log_to_validation=False, log_level=SPAM):
        if log_level >= PRINT_LOG_LEVEL:
            print("{}: {:20s} = {}".format(log_level, tag, value))
        if log_to_validation:
            self.validation_writer.add_scalar(tag, value,
                                              global_step=self.global_step)
        else:
            self.writer.add_scalar(tag, value, global_step=self.global_step)

    def save_images(self, tag, images, log_to_validation=False, log_to_writer=True):
        imsize = images.shape[2]
        image_dir = self.image_dir
        if log_to_validation:
            image_dir = os.path.join(self.image_dir, "validation")
        filename = "{0}{1}_{2}x{2}.jpg".format(tag, self.global_step, imsize)

        filepath = os.path.join(image_dir, filename)
        torchvision.utils.save_image(images, filepath, nrow=10)
        image_grid = torchvision.utils.make_grid(images, nrow=10)
        if log_to_writer:
            writer = self.validation_writer if log_to_validation else self.writer
            writer.add_image(tag, image_grid, self.global_step)
