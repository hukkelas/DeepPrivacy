import torch
from .build import CRITERION_REGISTRY
from typing import Dict, Tuple


class GanCriterion:

    NEED_REAL_SCORE_GENERATOR = False
    REQUIRES_D_SCORE = True

    def __init__(self, fake_index: int, *args, **kwargs):
        """
            fake_index: indicates which fake sample to use.
            Used in case for two-stage inpaintors (such as GatedConvolution)
        """
        self.fake_index = fake_index
        return

    def d_loss(self, batch: dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        return None

    def g_loss(self, batch: dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        return None


@CRITERION_REGISTRY.register_module
class WGANCriterion(GanCriterion):

    def d_loss(self, batch):
        real_scores = batch["real_scores"]
        fake_scores = batch["fake_scores"][self.fake_index]
        wasserstein_distance = 0
        for real, fake in zip(real_scores, fake_scores):
            wasserstein_distance += (real - fake)
        to_log = {
            "wasserstein_distance": wasserstein_distance.detach()
        }
        return (-wasserstein_distance).view(-1), to_log

    def g_loss(self, batch):
        fake_scores = batch["fake_scores"][self.fake_index]
        g_loss = 0
        for fake_score in fake_scores:
            g_loss -= fake_score
        g_loss = g_loss.view(-1)
        to_log = dict(
            g_loss=g_loss
        )
        return g_loss, to_log


@CRITERION_REGISTRY.register_module
class RGANCriterion(GanCriterion):

    NEED_REAL_SCORE_GENERATOR = True

    def __init__(self):
        super().__init__()
        self.bce_stable = torch.nn.BCEWithLogitsLoss(reduction="none")

    def d_loss(self, batch):
        real_scores = batch["real_scores"][:, 0]
        fake_scores = batch["fake_scores"][self.fake_index][:, 0]
        wasserstein_distance = (real_scores - fake_scores).squeeze()
        target = torch.ones_like(real_scores)
        d_loss = self.bce_stable(real_scores - fake_scores, target)
        to_log = {
            "wasserstein_distance": wasserstein_distance.mean().detach(),
            "d_loss": d_loss.mean().detach()
        }
        return d_loss.view(-1), to_log

    def g_loss(self, batch):
        real_scores = batch["real_scores"][:, 0]
        fake_scores = batch["fake_scores"][self.fake_index][:, 0]
        target = torch.ones_like(real_scores)
        g_loss = self.bce_stable(fake_scores - real_scores, target)
        to_log = dict(
            g_loss=g_loss.mean()
        )
        return g_loss.view(-1), to_log


@CRITERION_REGISTRY.register_module
class RaGANCriterion(GanCriterion):

    NEED_REAL_SCORE_GENERATOR = True

    def __init__(self):
        super().__init__()
        self.bce_stable = torch.nn.BCEWithLogitsLoss(reduction="none")

    def d_loss(self, batch):
        real_scores = batch["real_scores"][:, 0]
        fake_scores = batch["fake_scores"][self.fake_index][:, 0]
        wasserstein_distance = (real_scores - fake_scores).squeeze()
        target = torch.ones_like(real_scores)
        target2 = torch.zeros_like(real_scores)
        d_loss = self.bce_stable(real_scores - fake_scores.mean(), target) + \
            self.bce_stable(fake_scores - real_scores.mean(), target2)
        to_log = {
            "wasserstein_distance": wasserstein_distance.mean().detach(),
            "d_loss": d_loss.mean().detach()
        }
        return (d_loss / 2).view(-1), to_log

    def g_loss(self, batch):
        real_scores = batch["real_scores"][:, 0]
        fake_scores = batch["fake_scores"][self.fake_index][:, 0]
        target = torch.ones_like(real_scores)
        target2 = torch.zeros_like(real_scores)
        g_loss = self.bce_stable(real_scores - fake_scores.mean(), target2) + \
            self.bce_stable(fake_scores - real_scores.mean(), target)
        to_log = dict(
            g_loss=g_loss.mean()
        )
        return (g_loss / 2).view(-1), to_log


@CRITERION_REGISTRY.register_module
class NonSaturatingCriterion(GanCriterion):

    def d_loss(self, batch):
        real_scores = batch["real_scores"][:, 0]
        fake_scores = batch["fake_scores"][self.fake_index][:, 0]
        wasserstein_distance = (real_scores - fake_scores).squeeze()
        loss = torch.nn.functional.softplus(-real_scores) \
            + torch.nn.functional.softplus(fake_scores)
        return loss.view(-1), dict(
            wasserstein_distance=wasserstein_distance.mean().detach(),
            d_loss=loss.mean()
        )

    def g_loss(self, batch):
        fake_scores = batch["fake_scores"][self.fake_index][:, 0]
        loss = torch.nn.functional.softplus(-fake_scores).mean()
        return loss.view(-1), dict(
            g_loss=loss
        )
