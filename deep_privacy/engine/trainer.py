import torch
from deep_privacy import torch_utils
from deep_privacy import logger
from deep_privacy.dataset import build_dataloader_train, build_dataloader_val
from deep_privacy.modeling import loss, models
from .base_trainer import BaseTrainer
from .hooks import build_hooks


class Trainer(BaseTrainer):

    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__(cfg.output_dir)

        build_hooks(cfg, self)
        self.global_step = 0
        logger.init(cfg.output_dir)
        self.init_models()
        self.init_optimizer()
        self.load_checkpoint()

    def state_dict(self):
        state_dict = {
            "D": self.discriminator.state_dict(),
            "G": self.generator.state_dict(),
            "optimizer": self.loss_optimizer.state_dict(),
            "global_step": self.global_step,
            "running_average_generator": self.RA_generator.state_dict()
        }
        state_dict.update(super().state_dict())
        return state_dict

    def load_state_dict(self, state_dict: dict):
        self.global_step = state_dict["global_step"]
        logger.update_global_step(self.global_step)
        self.discriminator.load_state_dict(state_dict["D"])
        self.generator.load_state_dict(state_dict["G"])
        self.RA_generator.load_state_dict(
            state_dict["running_average_generator"]
        )
        self.RA_generator = torch_utils.to_cuda(self.RA_generator)
        self.init_optimizer()
        self.loss_optimizer.load_state_dict(state_dict["optimizer"])
        super().load_state_dict(state_dict)

    def batch_size(self) -> int:
        batch_size_schedule = self.cfg.trainer.batch_size_schedule
        return batch_size_schedule[self.current_imsize()]

    def init_models(self):
        self.discriminator = models.build_discriminator(
            self.cfg, data_parallel=torch.cuda.device_count() > 1)
        self.generator = models.build_generator(
            self.cfg, data_parallel=torch.cuda.device_count() > 1)
        self.RA_generator = models.build_generator(
            self.cfg, data_parallel=torch.cuda.device_count() > 1)
        self.RA_generator = torch_utils.to_cuda(self.RA_generator)
        self.RA_generator.load_state_dict(self.generator.state_dict())
        logger.info(str(self.generator))
        logger.info(str(self.discriminator))
        logger.log_variable(
            "stats/discriminator_parameters",
            torch_utils.number_of_parameters(self.discriminator))
        logger.log_variable(
            "stats/generator_parameters",
            torch_utils.number_of_parameters(self.generator))

    def current_imsize(self) -> int:
        return self.generator.current_imsize

    def train_step(self):
        self.before_step()
        batch = next(self.dataloader_train)
        logger.update_global_step(self.global_step)
        to_log = self.loss_optimizer.step(batch)
        while to_log is None:
            to_log = self.loss_optimizer.step(batch)
            self.hooks["StatsLogger"].num_skipped_steps += 1
        to_log = {f"loss/{key}": item for key, item in to_log.items()}
        self.hooks["StatsLogger"].to_log = to_log
        self.after_step()
        self.global_step += self.batch_size()

    def load_dataset(self):
        self.dataloader_train = iter(build_dataloader_train(
            self.cfg,
            self.current_imsize(),
            None))
        self.dataloader_val = build_dataloader_val(
            self.cfg,
            self.current_imsize(),
            None)

    def init_optimizer(self):
        self.loss_optimizer = loss.LossOptimizer.build_from_cfg(
            self.cfg, self.discriminator, self.generator
        )
        self.generator, self.discriminator = self.loss_optimizer.initialize_amp()
        logger.log_variable(
            "stats/learning_rate", self.loss_optimizer._learning_rate)


    def before_train(self):
        self.load_dataset()
        super().before_train()

    def before_step(self):
        logger.update_global_step(self.global_step)
        super().before_step()

    def train(self):
        self.before_train()
        while True:
            self.train_step()
