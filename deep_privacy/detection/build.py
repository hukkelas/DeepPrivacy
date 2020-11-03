from deep_privacy.utils import build_from_cfg, Registry


DETECTOR_REGISTRY = Registry("DETECTOR_REGISTRY")


def build_detector(cfg, *args, **kwargs):
    print(cfg)
    return build_from_cfg(cfg, DETECTOR_REGISTRY, *args, **kwargs)
