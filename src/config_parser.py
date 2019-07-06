from argparse import ArgumentParser
import os
import math
import json
import torch
import yaml
from collections import namedtuple

def convert_config(name, config):
    for key, value in config.items():
        if isinstance(value, dict) and key != "batch_size_schedule":

            config[key] = convert_config(key, value)
    return namedtuple(name, config.keys())(*config.values())

def load_config(config_path):
    with open(config_path, "r") as cfg_file:
        config = yaml.safe_load(cfg_file)
    return convert_config("Config", config)

def validate_start_channel_size(max_imsize, start_channel_size):
    # Assert start channel size is valid with the max imsize
    # Number of times to double
    n_image_double =  math.log(max_imsize, 2) - 2 # starts at 4
    n_channel_halving = math.log(start_channel_size, 2) + 2
    assert n_image_double < n_channel_halving

def print_config(dic, namespace="", first=False):
    if first:
        print("CONFIG USED:")
        print("="*80)
    dictionary = dic._asdict()
    #banned_keys = ["G", "D", "g_optimizer", "d_optimizer", "z_sample", "running_average_generator"]
    for (key, item) in dictionary.items():
        if first:
            new_namespace = key
        else:
            new_namespace = "{}.{}".format(namespace, key)
        
        if "_asdict" in dir(item):
            print_config(item, new_namespace, False)
        else:
            print("{:<50} {}".format(new_namespace, item))
        #print("{:<16} {}".format(key, item))
    if first:
        print("="*80)

def validate_config(config):
    assert config.train_config.amp_opt_level in ["O1","O0"], "Optimization level not correct. It was: {}".format(config.opt_level)
    validate_start_channel_size(config.max_imsize, config.models.start_channel_size)

def initialize_and_validate_config():
    parser = ArgumentParser()
    parser.add_argument("config_path",
                        help="Set the name of the model")

    args = parser.parse_args()
    assert os.path.isfile(args.config_path), "Did not find config file:".format(args.config_path)

    config = load_config(args.config_path)
    
    config_dir = os.path.dirname(args.config_path)
    
    new_config_fields = {
        "config_path": args.config_path,
        "checkpoint_dir": os.path.join(config_dir, "checkpoints"),
        "generated_data_dir": os.path.join(config_dir, "generated_data"),
        "summaries_dir": os.path.join(config_dir, "summaries")
    }
    config = namedtuple("Config", list(config._asdict().keys()) + list(new_config_fields.keys()))(
        *(list(config._asdict().values()) + list(new_config_fields.values()))
    )

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.generated_data_dir, exist_ok=True)

    validate_config(config)

    print_config(config, first=True)
    return config


if __name__ == "__main__":
    config = load_config("../models/default/config.yml")
    print_config(config, first=True)
    initialize_and_validate_config()