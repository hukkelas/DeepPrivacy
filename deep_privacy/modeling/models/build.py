from deep_privacy.utils import build_from_cfg, Registry
from .utils import NetworkWrapper

DISCRIMINATOR_REGISTRY = Registry("DISCRIMINATOR_REGISTRY")
GENERATOR_REGISTRY = Registry("GENERATOR_REGISTRY")


def build_discriminator(
        cfg,
        data_parallel):
    discriminator = build_from_cfg(
        cfg.models.discriminator, DISCRIMINATOR_REGISTRY,
        cfg=cfg,
        max_imsize=cfg.models.max_imsize,
        pose_size=cfg.models.pose_size,
        image_channels=cfg.models.image_channels,
        conv_size=cfg.models.conv_size)
    if data_parallel:
        discriminator = NetworkWrapper(discriminator)
    discriminator = extend_model(cfg, discriminator)
    return discriminator


def build_generator(
        cfg,
        data_parallel):
    generator = build_from_cfg(
        cfg.models.generator, GENERATOR_REGISTRY,
        cfg=cfg,
        max_imsize=cfg.models.max_imsize,
        conv_size=cfg.models.conv_size,
        image_channels=cfg.models.image_channels,
        pose_size=cfg.models.pose_size)
    if data_parallel:
        generator = NetworkWrapper(generator)
    generator = extend_model(cfg, generator)
    return generator


def extend_model(cfg, model):
    while model.current_imsize < cfg.models.min_imsize:
        model.extend()
    if cfg.trainer.progressive.enabled:
        return model
    while model.current_imsize < cfg.models.max_imsize:
        model.extend()
    return model
