import torch
from deep_privacy import logger
import pathlib


def _get_map_location():
    if not torch.cuda.is_available():
        logger.warn(
            "Cuda is not available. Forcing map checkpoint to be loaded into CPU.")
        return "cpu"
    return None


def load_checkpoint_from_url(model_url: str):
    if model_url is None:
        return None
    return torch.hub.load_state_dict_from_url(
        model_url, map_location=_get_map_location())


def load_checkpoint(ckpt_dir_or_file: pathlib.Path) -> dict:
    if ckpt_dir_or_file.is_dir():
        with open(ckpt_dir_or_file.joinpath('latest_checkpoint')) as f:
            ckpt_path = f.readline().strip()
            ckpt_path = ckpt_dir_or_file.joinpath(ckpt_path)
    else:
        ckpt_path = ckpt_dir_or_file
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Did not find path: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=_get_map_location())
    logger.info(f"Loaded checkpoint from {ckpt_path}")
    return ckpt


def _get_checkpoint_path(
        output_dir: str, validation_checkpoint_step: int = None):
    if validation_checkpoint_step is None:
        return pathlib.Path(output_dir, "checkpoints")
    step = validation_checkpoint_step * 10**6
    path = pathlib.Path(
        output_dir, "validation_checkpoints", f"step_{step}.ckpt")
    return path


def get_checkpoint(
        output_dir: str, validation_checkpoint_step: int = None):
    path = _get_checkpoint_path(output_dir, validation_checkpoint_step)
    return load_checkpoint(path)


def get_previous_checkpoints(directory: pathlib.Path) -> list:
    if directory.is_file():
        directory = directory.parent
    list_path = directory.joinpath("latest_checkpoint")
    list_path.touch(exist_ok=True)
    with open(list_path) as fp:
        ckpt_list = fp.readlines()
    return [_.strip() for _ in ckpt_list]


def get_checkpoint_step(output_dir: str, validation_checkpoint_step: int):
    if validation_checkpoint_step is not None:
        return validation_checkpoint_step
    directory = _get_checkpoint_path(output_dir)
    ckpt_path = get_previous_checkpoints(directory)[0]
    print(ckpt_path)
    ckpt_path = pathlib.Path(ckpt_path)
    step = ckpt_path.stem.replace("step_", "")
    step = step.replace(".ckpt", "")
    return int(step)


class Checkpointer:

    def __init__(self, output_dir: str):
        self.checkpoint_dir = pathlib.Path(
            output_dir, "checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def save_checkpoint(
            self,
            state_dict: dict,
            filepath: pathlib.Path = None,
            max_keep=2):
        if filepath is None:
            global_step = self.trainer.global_step
            filename = f"step_{global_step}.ckpt"
            filepath = self.checkpoint_dir.joinpath(filename)
        list_path = filepath.parent.joinpath("latest_checkpoint")
        torch.save(state_dict, filepath)
        previous_checkpoints = get_previous_checkpoints(filepath)
        if filepath.name not in previous_checkpoints:
            previous_checkpoints = [filepath.name] + previous_checkpoints
        if len(previous_checkpoints) > max_keep:
            for ckpt in previous_checkpoints[max_keep:]:
                path = self.checkpoint_dir.joinpath(ckpt)
                if path.exists():
                    logger.info(f"Removing old checkpoint: {path}")
                    path.unlink()
        previous_checkpoints = previous_checkpoints[:max_keep]
        with open(list_path, 'w') as fp:
            fp.write("\n".join(previous_checkpoints))
        logger.info(f"Saved checkpoint to: {filepath}")

    def checkpoint_exists(self) -> bool:
        num_checkpoints = len(list(self.checkpoint_dir.glob("*.ckpt")))
        return num_checkpoints > 0

    def load_checkpoint(self) -> dict:
        checkpoint = load_checkpoint(self.checkpoint_dir)
        return checkpoint
