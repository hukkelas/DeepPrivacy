import signal
import pathlib
from deep_privacy import logger
from .base import HookBase, HOOK_REGISTRY


@HOOK_REGISTRY.register_module
class RunningAverageHook(HookBase):

    def before_train(self):
        self.update_beta()

    def update_beta(self):
        batch_size = self.trainer.batch_size()
        g = self.trainer.RA_generator
        g.update_beta(
            batch_size
        )
        logger.log_variable("stats/running_average_decay", g.ra_beta)

    def before_extend(self):
        self.update_beta()

    def after_extend(self):
        self.update_beta()

    def after_step(self):
        rae_generator = self.trainer.RA_generator
        generator = self.trainer.generator
        rae_generator.update_ra(generator)


@HOOK_REGISTRY.register_module
class CheckpointHook(HookBase):

    def __init__(
            self,
            ims_per_checkpoint: int,
            output_dir: pathlib.Path):
        self.ims_per_checkpoint = ims_per_checkpoint
        self.next_validation_checkpoint = ims_per_checkpoint
        self.validation_checkpoint_dir = pathlib.Path(
            output_dir, "validation_checkpoints")
        self.transition_checkpoint_dir = pathlib.Path(
            output_dir, "transition_checkpoints")
        self.validation_checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.transition_checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def after_step(self):
        if self.global_step() >= self.next_validation_checkpoint:
            self.next_validation_checkpoint += self.ims_per_checkpoint
            self.trainer.save_checkpoint()

        self.save_validation_checkpoint()

    def state_dict(self):
        return {"next_validation_checkpoint": self.next_validation_checkpoint}

    def load_state_dict(self, state_dict: dict):
        next_validation_checkpoint = state_dict["next_validation_checkpoint"]
        self.next_validation_checkpoint = next_validation_checkpoint

    def save_validation_checkpoint(self):
        checkpoints = [12, 20, 30, 40, 50]
        for checkpoint_step in checkpoints:
            checkpoint_step = checkpoint_step * 10**6
            previous_global_step = self.global_step() - self.trainer.batch_size()
            if self.global_step() >= checkpoint_step and previous_global_step < checkpoint_step:
                logger.info("Saving global checkpoint for validation")
                filepath = self.validation_checkpoint_dir.joinpath(
                    f"step_{self.global_step()}.ckpt"
                )
                self.trainer.save_checkpoint(
                    filepath, max_keep=len(checkpoints) + 1)

    def before_extend(self):
        filepath = self.transition_checkpoint_dir.joinpath(
            f"imsize_{self.current_imsize()}.ckpt"
        )
        self.trainer.save_checkpoint(filepath)


@HOOK_REGISTRY.register_module
class SigTermHook(HookBase):

    def __init__(self):
        self.sigterm_received = False
        signal.signal(signal.SIGINT, self.handle_sigterm)
        signal.signal(signal.SIGTERM, self.handle_sigterm)

    def handle_sigterm(self, signum, frame):
        logger.info(
            "[SIGTERM RECEVIED] Received sigterm. Stopping train after step.")
        self.sigterm_received = True
        exit()

    def after_step(self):
        if self.sigterm_received:
            logger.info("[SIGTERM RECEIVED] Stopping train.")
            self.trainer.save_checkpoint(max_keep=3)
            exit()
