import pathlib
import torch
import os
from urllib.parse import urlparse
from deep_privacy import logger, torch_utils
from deep_privacy.config import Config
from deep_privacy.inference.infer import load_model_from_checkpoint
from deep_privacy.inference.deep_privacy_anonymizer import DeepPrivacyAnonymizer

available_models = [
    "fdf128_rcnn512",
    "fdf128_retinanet512",
    "fdf128_retinanet256",
    "fdf128_retinanet128",
    "deep_privacy_V1",

]

config_urls = {
    "fdf128_retinanet512": "https://folk.ntnu.no/haakohu/configs/fdf/retinanet512.json",
    "fdf128_retinanet256": "https://folk.ntnu.no/haakohu/configs/fdf/retinanet256.json",
    "fdf128_retinanet128": "https://folk.ntnu.no/haakohu/configs/fdf/retinanet128.json",
    "fdf128_rcnn512": "https://folk.ntnu.no/haakohu/configs/fdf_512.json",
    "deep_privacy_V1": "https://folk.ntnu.no/haakohu/configs/deep_privacy_v1.json",
}


def get_config(config_url):
    parts = urlparse(config_url)
    cfg_name = os.path.basename(parts.path)
    assert cfg_name is not None
    cfg_path = pathlib.Path(
        torch.hub._get_torch_home(), "deep_privacy_cache", cfg_name)
    cfg_path.parent.mkdir(exist_ok=True, parents=True)
    if not cfg_path.is_file():
        torch.hub.download_url_to_file(config_url, cfg_path)
    assert cfg_path.is_file()
    return Config.fromfile(cfg_path)


def build_anonymizer(
        model_name=available_models[0],
        batch_size: int = 1,
        fp16_inference: bool = True,
        truncation_level: float = 0,
        detection_threshold: float = .1,
        opts: str = None,
        config_path: str = None,
        return_cfg=False) -> DeepPrivacyAnonymizer:
    """
        Builds anonymizer with detector and generator from checkpoints.

        Args:
            config_path: If not None, will override model_name
            opts: if not None, can override default settings. For example:
                opts="anonymizer.truncation_level=5, anonymizer.batch_size=32"
    """
    if config_path is None:
        print(config_path)
        assert model_name in available_models,\
            f"{model_name} not in available models: {available_models}"
        cfg = get_config(config_urls[model_name])
    else:
        cfg = Config.fromfile(config_path)
    logger.info("Loaded model:" + cfg.model_name)
    generator = load_model_from_checkpoint(cfg)
    logger.info(f"Generator initialized with {torch_utils.number_of_parameters(generator)/1e6:.2f}M parameters")
    cfg.anonymizer.truncation_level = truncation_level
    cfg.anonymizer.batch_size = batch_size
    cfg.anonymizer.fp16_inference = fp16_inference
    cfg.anonymizer.detector_cfg.face_detector_cfg.confidence_threshold = detection_threshold
    cfg.merge_from_str(opts)
    anonymizer = DeepPrivacyAnonymizer(generator, cfg=cfg, **cfg.anonymizer)
    if return_cfg:
        return anonymizer, cfg
    return anonymizer
