import torchvision
import pathlib
import math
import logging
from . import torch_utils
from torch.utils.tensorboard import SummaryWriter


writer = None
global_step = 0
image_dir = None
logFormatter = logging.Formatter(
    "%(asctime)s [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)


def init(output_dir):
    global writer, image_dir, rootLogger
    logdir = pathlib.Path(
        output_dir, "summaries")
    writer = SummaryWriter(logdir.joinpath("train"))

    image_dir = pathlib.Path(output_dir, "generated_data")
    image_dir.joinpath("validation").mkdir(exist_ok=True, parents=True)
    image_dir.joinpath("transition").mkdir(exist_ok=True, parents=True)
    filepath = pathlib.Path(output_dir, "train.log")
    fileHandler = logging.FileHandler(filepath)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)


def update_global_step(val):
    global global_step
    global_step = val


def log_variable(tag, value, log_to_validation=False, log_level=logging.DEBUG):
    if math.isnan(value):
        rootLogger.debug(f"Tried to log nan/inf for tag={tag}")
        return
    value = float(value)
    rootLogger.log(log_level, f"{tag}: {value}")
    assert not log_to_validation
    writer.add_scalar(tag, value, global_step=global_step)


def log_dictionary(dictionary: dict, log_to_validation=False):
    for key, item in dictionary.items():
        log_variable(key, item, log_to_validation=log_to_validation)


def save_images(tag, images,
                log_to_validation=False,
                log_to_writer=True,
                nrow=10,
                denormalize=False):
    if denormalize:
        images = torch_utils.denormalize_img(images)
    imsize = images.shape[2]
    imdir = image_dir
    if log_to_validation:
        imdir = image_dir.joinpath("validation")
    filename = "{0}{1}_{2}x{2}.jpg".format(tag, global_step, imsize)

    filepath = imdir.joinpath(filename)
    torchvision.utils.save_image(images, filepath, nrow=nrow)
    image_grid = torchvision.utils.make_grid(images, nrow=nrow)
    if log_to_writer:
        if log_to_validation:
            tag = f"validation/{tag}"
        else:
            tag = f"train/{tag}"
        writer.add_image(tag, image_grid, global_step)


def info(text):
    rootLogger.info(text)


def warn(text):
    rootLogger.warn(text)
