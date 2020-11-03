import torch
import numpy as np
import weakref
import collections
from .checkpointer import Checkpointer
from . import hooks
from deep_privacy import logger


torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = True


class BaseTrainer:

    def __init__(self, output_dir: str):
        self.hooks: collections.OrderedDict[str, hooks.HookBase] = {}
        self.sigterm_received = False
        self.checkpointer = Checkpointer(output_dir)
        self.checkpointer.trainer = self

    def register_hook(self, key: str, hook: hooks.HookBase):
        assert key not in self.hooks
        self.hooks[key] = hook
        # To avoid circular reference, hooks and trainer cannot own each other.
        # This normally does not matter, but will cause memory leak if the
        # involved objects contain __del__:
        # See
        # http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
        assert isinstance(hook, hooks.HookBase)
        hook.trainer = weakref.proxy(self)

    def before_extend(self):
        for hook in self.hooks.values():
            hook.before_extend()

    def before_train(self):
        for hook in self.hooks.values():
            hook.before_train()

    def before_step(self):
        for hook in self.hooks.values():
            hook.before_step()

    def after_step(self):
        for hook in self.hooks.values():
            hook.after_step()

    def after_extend(self):
        for hook in self.hooks.values():
            hook.after_extend()

    def state_dict(self) -> dict:
        state_dict = {}
        for key, hook in self.hooks.items():
            hsd = hook.state_dict()
            if hsd is not None:
                state_dict[key] = hook.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict):
        for key, hook in self.hooks.items():
            if hook.state_dict() is None:
                continue
            hook.load_state_dict(state_dict[key])

    def save_checkpoint(self, filepath=None, max_keep=2):
        logger.info(f"Saving checkpoint to: {filepath}")
        state_dict = self.state_dict()
        self.checkpointer.save_checkpoint(
            state_dict, filepath, max_keep)

    def load_checkpoint(self):
        if not self.checkpointer.checkpoint_exists():
            return
        state_dict = self.checkpointer.load_checkpoint()
        self.load_state_dict(state_dict)
