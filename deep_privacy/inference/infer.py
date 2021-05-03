import argparse
import torch
import numpy as np
from deep_privacy.engine.checkpointer import get_checkpoint, load_checkpoint_from_url
from typing import List
from deep_privacy import config, logger
import deep_privacy.torch_utils as torch_utils
from deep_privacy import modeling


def init_generator(cfg, ckpt=None):
    g = modeling.models.build_generator(cfg, data_parallel=False)
    if ckpt is not None:
        g.load_state_dict(ckpt["running_average_generator"])
    g.eval()
    torch_utils.to_cuda(g)
    return g


def infer_parser() -> argparse.ArgumentParser:
    parser = config.default_parser()
    parser.add_argument(
        "-s", "--source_path",
        help="Target to infer",
        default="test_examples/images"
    )
    parser.add_argument(
        "-t", "--target_path",
        help="Target path to save anonymized result.\
                Defaults to subdirectory of config file."
    )
    parser.add_argument(
        "--step", default=None, type=int,
        help="Set validation checkpoint to load. Defaults to most recent"
    )
    return parser


def load_model_from_checkpoint(
        cfg,
        validation_checkpoint_step: int = None,
        include_discriminator=False):
    try:
        ckpt = get_checkpoint(cfg.output_dir, validation_checkpoint_step)
    except FileNotFoundError:
        cfg.model_url = f'{cfg.model_url}'.replace("http://", "https://", 1)
        ckpt = None
        ckpt = load_checkpoint_from_url(cfg.model_url)
    if ckpt is None:
        logger.warn(f"Could not find checkpoint. {cfg.output_dir}")
    generator = init_generator(cfg, ckpt)
    generator = jit_wrap(generator, cfg)
    if include_discriminator:
        discriminator = modeling.models.build_discriminator(
            cfg, data_parallel=False)
        discriminator.load_state_dict(ckpt["D"])
        discriminator = torch_utils.to_cuda(discriminator)
        return generator, discriminator
    return generator


def jit_wrap(generator, cfg):
    """
        Torch JIT wrapper for accelerated inference
    """
    if not cfg.anonymizer.jit_trace:
        return generator
    imsize = cfg.models.max_imsize
    x_in = torch.randn((1, 3, imsize, imsize))
    example_inp = dict(
        condition=x_in,
        mask=torch.randn((1, 1, imsize, imsize)).bool().float(),
        landmarks=torch.randn((1, cfg.models.pose_size)),
        z=truncated_z(x_in, cfg.models.generator.z_shape, 5)
    )
    example_inp = {k: torch_utils.to_cuda(v) for k, v in example_inp.items()}
    generator = torch.jit.trace(
        generator,
        (example_inp["condition"],
         example_inp["mask"],
         example_inp["landmarks"],
         example_inp["z"]),
        optimize=True)
    return generator


def truncated_z(x_in: torch.Tensor, z_shape, truncation_level: float):
    z_shape = ((x_in.shape[0], *z_shape))
    if truncation_level == 0:
        return torch.zeros(z_shape, dtype=x_in.dtype, device=x_in.device)

    z = torch.randn(z_shape, device=x_in.device, dtype=x_in.dtype)
    while z.abs().max() >= truncation_level:
        mask = z.abs() >= truncation_level
        z_ = torch.randn(z_shape, device=x_in.device, dtype=x_in.dtype)
        z[mask] = z_[mask]
    return z


def infer_images(
        dataloader, generator, truncation_level: float,
        verbose=False,
        return_condition=False) -> List[np.ndarray]:
    imshape = (generator.current_imsize, generator.current_imsize, 3)
    real_images = np.empty(
        (dataloader.num_images(), *imshape), dtype=np.float32)
    fake_images = np.empty_like(real_images)
    if return_condition:
        conditions = np.empty_like(fake_images)
    batch_size = dataloader.batch_size
    generator.eval()
    dl_iter = iter(dataloader)
    if verbose:
        import tqdm
        dl_iter = tqdm.tqdm(dl_iter)
    with torch.no_grad():
        for idx, batch in enumerate(dl_iter):
            real_data = batch["img"]
            z = truncated_z(real_data, generator.z_shape, truncation_level)
            fake_data = generator(**batch, z=z)
            start = idx * batch_size
            end = start + len(real_data)
            real_data = torch_utils.image_to_numpy(real_data, denormalize=True)
            fake_data = torch_utils.image_to_numpy(fake_data, denormalize=True)
            real_images[start:end] = real_data
            fake_images[start:end] = fake_data
            if return_condition:
                conditions[start:end] = torch_utils.image_to_numpy(
                    batch["condition"], denormalize=True)
    generator.train()
    if return_condition:
        return real_images, fake_images, conditions
    return real_images, fake_images
