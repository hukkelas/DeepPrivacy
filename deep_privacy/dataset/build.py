from deep_privacy.utils import Registry, build_from_cfg
from .utils import fast_collate, DataPrefetcher, progressive_decorator
from .transforms import build_transforms
import torch
DATASET_REGISTRY = Registry("DATASET")


def get_dataloader(cfg, imsize, get_transition_value, is_train):
    cfg_data = cfg.data_val
    if is_train:
        cfg_data = cfg.data_train
    if cfg_data.dataset.type == "MNISTDataset":
        assert cfg.models.pose_size == 0
    transform = build_transforms(cfg_data.transforms, imsize=imsize)
    dataset = build_from_cfg(
        cfg_data.dataset,
        DATASET_REGISTRY,
        imsize=imsize,
        transform=transform
    )
    batch_size = cfg.trainer.batch_size_schedule[imsize]
    dataloader = torch.utils.data.DataLoader(
        dataset,
        pin_memory=False,
        collate_fn=fast_collate,
        batch_size=batch_size,
        **cfg_data.loader
    )
    dataloader = DataPrefetcher(
        dataloader,
        infinite_loader=is_train
    )
    # If progressive growing, perform GPU image interpolation
    if not cfg.trainer.progressive.enabled:
        return dataloader
    if get_transition_value is not None:
        assert cfg.trainer.progressive.enabled
    dataloader.next = progressive_decorator(
        dataloader.next,
        get_transition_value)
    return dataloader


def build_dataloader_train(
        cfg,
        imsize,
        get_transition_value=None):
    return get_dataloader(
        cfg, imsize,
        get_transition_value,
        is_train=True
    )


def build_dataloader_val(
        cfg,
        imsize,
        get_transition_value=None):
    return get_dataloader(
        cfg, imsize,
        get_transition_value,
        is_train=False
    )


if __name__ == "__main__":
    import argparse
    from deep_privacy.config import Config
    from . import *
    from deep_privacy import torch_utils
    from deep_privacy.visualization.utils import draw_faces_with_keypoints, np_make_image_grid
    from PIL import Image
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    cfg = Config.fromfile(args.config_path)
    imsize = cfg.models.max_imsize
    dl_train = build_dataloader_val(cfg, imsize)
    batch = next(iter(dl_train))
    im0 = batch["condition"]
    im1 = batch["img"]

    im = torch.cat((im1, im0), axis=-1)
    im = torch_utils.image_to_numpy(
        im, to_uint8=True, denormalize=True)
    if "landmarks" in batch:
        lm = batch["landmarks"].cpu().numpy() * imsize
        lm = lm.reshape(lm.shape[0], -1, 2)
        im = [
            draw_faces_with_keypoints(_, None, im_keypoints=[lm[i]], radius=3)
            for i, _ in enumerate(im)
        ]
    print([_.shape for _ in im])
    im = np_make_image_grid(im, nrow=10)
    im = Image.fromarray(im)
    im.show()
    im.save("example.png")
