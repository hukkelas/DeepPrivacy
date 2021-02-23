import torch
import typing
import numpy as np
from deep_privacy import torch_utils
from deep_privacy.modeling import models
from deep_privacy.utils import build_from_cfg
from .build import CRITERION_REGISTRY
from typing import Tuple
from .loss import GradientPenalty, GanCriterion
try:
    from apex import amp
    from apex.optimizers import FusedAdam
except ImportError:
    pass


class LossOptimizer:

    def __init__(self,
                 discriminator: models.discriminator.Discriminator,
                 generator: models.generator.Generator,
                 criterions_D: typing.List[GanCriterion],
                 criterions_G: typing.List[GanCriterion],
                 learning_rate: float,
                 amp_opt_level: str,
                 lazy_regularization: bool):
        self.generator = generator
        self.discriminator = discriminator
        self.criterions_D = criterions_D
        self.criterions_G = criterions_G
        self.it = 0
        self._amp_opt_level = amp_opt_level
        self._lazy_regularization = lazy_regularization
        self._learning_rate = learning_rate
        self.init_optimizers()
        # For Two-Stage inpaintors we might want a discriminator score for
        # several outputs
        self.required_D_index = list(set([
            c.fake_index for c in criterions_D + criterions_G
            if c.REQUIRES_D_SCORE]))

    def state_dict(self):
        return {
            "d_optimizer": self.d_optimizer.state_dict(),
            "g_optimizer": self.g_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.d_optimizer.load_state_dict(state_dict["d_optimizer"])
        self.g_optimizer.load_state_dict(state_dict["g_optimizer"])

    @staticmethod
    def build_from_cfg(cfg, discriminator, generator):
        lazy_regularization = cfg.trainer.optimizer.lazy_regularization
        criterions_D = [
            build_from_cfg(
                criterion, CRITERION_REGISTRY,
                discriminator=discriminator,
                lazy_regularization=lazy_regularization)
            for criterion in cfg.discriminator_criterions.values()
            if criterion is not None
        ]
        criterions_G = [
            build_from_cfg(
                criterion, CRITERION_REGISTRY, discriminator=discriminator)
            for criterion in cfg.generator_criterions.values()
            if criterion is not None
        ]
        return LossOptimizer(
            discriminator, generator, criterions_D, criterions_G,
            **cfg.trainer.optimizer)

    def init_optimizers(self) -> Tuple[torch.nn.Module]:
        torch_utils.to_cuda(
            [self.generator, self.discriminator])
        betas_d = (0.0, 0.99)
        lr_d = self._learning_rate
        if self._lazy_regularization:
            lazy_interval = [
                criterion.lazy_reg_interval
                for criterion in self.criterions_D
                if isinstance(criterion, GradientPenalty)]
            assert len(lazy_interval) <= 1
            if len(lazy_interval) == 1:
                lazy_interval = lazy_interval[0]
                c = lazy_interval / (lazy_interval + 1)
                betas_d = [beta ** c for beta in betas_d]
                lr_d *= c

        self.d_optimizer = FusedAdam(self.discriminator.parameters(),
                                     lr=lr_d,
                                     betas=betas_d)
        self.g_optimizer = FusedAdam(self.generator.parameters(),
                                     lr=self._learning_rate,
                                     betas=(0.0, 0.99))

    def initialize_amp(self):
        """
            Have to call initialize AMP from trainer since it changes the reference to generator / discriminator?
        """
        [self.generator, self.discriminator], [self.g_optimizer, self.d_optimizer] = amp.initialize(
            [self.generator, self.discriminator],
            [self.g_optimizer, self.d_optimizer],
            opt_level=self._amp_opt_level,
            num_losses=len(self.criterions_D) + len(self.criterions_G),
            max_loss_scale=2.**17,
        )
        return self.generator, self.discriminator

    def step(self, batch):
        losses_d = self.step_D(batch)
        losses_g = self.step_G(batch)
        if losses_d is None or losses_g is None:
            return None
        self.it += 1
        losses = {**losses_d, **losses_g}
        return losses

    def _backward(self, batch, loss_funcs, model, optimizer, id_offset):
        log = {}
        for param in model.parameters():
            param.grad = None
        for i, loss_fnc in enumerate(loss_funcs):
            loss, to_log = loss_fnc(batch)
            if loss is None:
                continue
            log.update(to_log)
            retain_graph = len(loss_funcs) - 1 != i
            l_id = id_offset + i
            loss = loss.mean()
            with amp.scale_loss(loss, optimizer, loss_id=l_id) as scaled_loss:
                scaled_loss.backward(retain_graph=retain_graph)
        optimizer.step()
        return {key: item.mean().detach() for key, item in log.items()}

    def step_D(self, batch):
        # Forward discriminator
        if len(self.required_D_index) == 0:
            return {}
        with torch.no_grad():
            fake_data = self.generator.forward_train(**batch)
        real_scores = self.discriminator(
            **batch, with_pose=True)

        fake_scores = {}
        for idx in self.required_D_index:
            fake_scores[idx] = self.discriminator.forward_fake(
                **batch, with_pose=True, fake_img=fake_data[idx])

        batch = {key: item for key, item in batch.items()}
        batch["fake_data"] = fake_data
        batch["real_scores"] = real_scores
        batch["fake_scores"] = fake_scores
        # Backward
        loss_funcs = [c.d_loss for c in self.criterions_D]
        log = self._backward(
            batch,
            loss_funcs, self.discriminator,
            self.d_optimizer,
            id_offset=0
        )
        for i in range(len(real_scores)):
            log[f"real_score{i}"] = real_scores[i].mean().detach()
        for _, fake_score in fake_scores.items():
            log[f"fake_score{i}"] = fake_score.mean().detach()
        return log

    def step_G(self, batch):
        for p in self.discriminator.parameters():
            p.requires_grad = False
        # Forward
        fake_data = self.generator.forward_train(**batch)
        fake_scores = {}
        for idx in self.required_D_index:
            fake_scores[idx] = self.discriminator.forward_fake(
                **batch, fake_img=fake_data[idx])
        batch = {key: item for key, item in batch.items()}
        batch["fake_data"] = fake_data
        batch["fake_scores"] = fake_scores
        self.mask = batch["mask"]
        if any(c.NEED_REAL_SCORE_GENERATOR for c in self.criterions_G):
            real_scores = self.discriminator(**batch)
            batch["real_scores"] = real_scores
        loss_funcs = [c.g_loss for c in self.criterions_G]
        log = self._backward(
            batch,
            loss_funcs,
            self.generator,
            self.g_optimizer,
            id_offset=len(self.criterions_D),
        )
        del self.mask
        for p in self.discriminator.parameters():
            p.requires_grad = True
        return log
