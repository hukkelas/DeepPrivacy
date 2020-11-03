import numpy as np
from deep_privacy import torch_utils, logger
from .trainer import Trainer
from deep_privacy.dataset import build_dataloader_train, build_dataloader_val


class ProgressiveTrainer(Trainer):

    def __init__(self, cfg):
        self.prev_transition = 0
        self.transition_iters = cfg.trainer.progressive.transition_iters
        self.transition_value = None
        super().__init__(cfg)

    def state_dict(self) -> dict:
        state_dict = super().state_dict()
        state_dict.update({
            "prev_transition": self.prev_transition
        })
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.prev_transition = state_dict["prev_transition"]

    def _grow_phase(self):
        # Log transition value here to not create misguiding representation on
        # tensorboard
        if self.transition_value is not None:
            logger.log_variable(
                "stats/transition-value", self.get_transition_value())

        self._update_transition_value()
        transition_iters = self.transition_iters
        minibatch_repeats = self.cfg.trainer.progressive.minibatch_repeats
        next_transition = self.prev_transition + transition_iters
        num_batches = (next_transition - self.global_step) / self.batch_size()
        num_batches = int(np.ceil(num_batches))
        num_repeats = int(np.ceil(num_batches / minibatch_repeats))
        logger.info(
            f"Starting grow phase for imsize={self.current_imsize()}" +
            f" Training for {num_batches} batches with batch size: {self.batch_size()}")
        for it in range(num_repeats):
            for _ in range(min(minibatch_repeats,
                               num_batches - it * minibatch_repeats)):
                self.train_step()
            self._update_transition_value()
        # Check that grow phase happens at correct spot
        assert self.global_step >= self.prev_transition + transition_iters,\
            f"Global step: {self.global_step}, batch size: {self.batch_size()}, prev_transition: {self.prev_transition}" +\
            f" transition iters: {transition_iters}"
        assert self.global_step - self.batch_size() <= self.prev_transition + transition_iters,\
            f"Global step: {self.global_step}, batch size: {self.batch_size()}, prev_transition: {self.prev_transition}" +\
            f" transition iters: {transition_iters}"

    def _update_transition_value(self):
        if self._get_phase() == "stability":
            self.transition_value = 1.0
        else:
            remaining = self.global_step - self.prev_transition
            v = remaining / self.transition_iters
            assert 0 <= v <= 1
            self.transition_value = v
        self.generator.update_transition_value(self.transition_value)
        self.discriminator.update_transition_value(self.transition_value)
        self.RA_generator.update_transition_value(self.transition_value)
        logger.log_variable(
            "stats/transition-value", self.get_transition_value())

    def get_transition_value(self):
        return self.transition_value

    def _stability_phase(self):
        self._update_transition_value()
        assert self.get_transition_value() == 1.0

        if self.prev_transition == 0:
            next_transition = self.transition_iters
        else:
            next_transition = self.prev_transition + self.transition_iters * 2

        num_batches = (next_transition - self.global_step) / self.batch_size()
        num_batches = int(np.ceil(num_batches))
        assert num_batches > 0
        logger.info(
            f"Starting stability phase for imsize={self.current_imsize()}" +
            f" Training for {num_batches} batches with batch size: {self.batch_size()}")
        for it in range(num_batches):
            self.train_step()

    def _get_phase(self):
        # Initial training pahse
        if self.global_step < self.transition_iters:
            return "stability"
        # Last phase
        if self.current_imsize() == self.cfg.models.max_imsize:
            if self.global_step >= self.prev_transition + self.transition_iters:
                return "stability"
            return "grow"
        if self.global_step < self.prev_transition + self.transition_iters:
            return "grow"
        assert self.prev_transition + self.transition_iters <= self.global_step
        assert self.global_step <= self.prev_transition + self.transition_iters * 2
        return "stability"

    def train_infinite(self):
        self._update_transition_value()
        while True:
            self.train_step()

    def train(self):
        self.before_train()
        while self.current_imsize() != self.cfg.models.max_imsize:
            if self._get_phase() == "grow":
                self._grow_phase()
            else:
                self._stability_phase()
                self.grow_models()
                self.prev_transition = self.global_step
        else:
            if self._get_phase() == "grow":
                self._grow_phase()
        self.train_infinite()

    def grow_models(self):
        self.before_extend()
        self.discriminator.extend()
        self.generator.extend()
        self.RA_generator.extend()
        self.RA_generator = torch_utils.to_cuda(self.RA_generator)
        del self.dataloader_train, self.dataloader_val
        self.load_dataset()
        self.init_optimizer()
        self.after_extend()

    def load_dataset(self):
        self.dataloader_train = iter(build_dataloader_train(
            self.cfg,
            self.current_imsize(),
            self.get_transition_value))
        self.dataloader_val = build_dataloader_val(
            self.cfg,
            self.current_imsize(),
            self.get_transition_value)
