import torch
from .build import CRITERION_REGISTRY
from deep_privacy.modeling import models
from .adversarial_loss import GanCriterion


@CRITERION_REGISTRY.register_module
class GradientPenalty(GanCriterion):

    def __init__(self,
                 lambd: float,
                 mask_region_only: bool,
                 norm: str,
                 distance: str,
                 discriminator,
                 lazy_regularization: bool,
                 lazy_reg_interval: int,
                 mask_decoder_gradient: bool,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discriminator = discriminator
        if mask_decoder_gradient:
            assert isinstance(discriminator, models.UNetDiscriminator)
        self._mask_decoder_gradient = mask_decoder_gradient
        self.lazy_reg_interval = lazy_reg_interval
        self.mask_region_only = mask_region_only
        self._norm = norm
        self._lambd = lambd
        if lazy_regularization:
            self._lambd *= lazy_reg_interval
        else:
            self.lazy_reg_interval = 1
        self._distance = distance
        assert self._norm in ["L2", "Linf"]
        self.it = 0

    def clip(self, activation):
        if self._distance == "clamp":
            return torch.nn.functional.relu(activation)
        assert self._distance == "L2"
        return activation.pow(2)

    def norm(self, grad):
        if self._norm == "L2":  # L2 Norm
            grad_norm = grad.norm(p=2, dim=1)
        else:  # Linfinity norm
            grad_abs = grad.abs()
            grad_norm, _ = torch.max(grad_abs, dim=1)
        return grad_norm

    def d_loss(self, batch):
        self.it += 1
        if self.it % self.lazy_reg_interval != 0:
            return None, None
        real_data = batch["img"]
        fake_data = batch["fake_data"][self.fake_index]
        mask = batch["mask"]
        epsilon_shape = [real_data.shape[0]] + [1] * (real_data.dim() - 1)
        epsilon = torch.rand(epsilon_shape)
        epsilon = epsilon.to(fake_data.device, fake_data.dtype)
        real_data = real_data.to(fake_data.dtype)
        x_hat = epsilon * real_data + (1 - epsilon) * fake_data.detach()
        x_hat.requires_grad = True
        logits = self.discriminator.forward_fake(**batch, fake_img=x_hat)
        to_backward = 0
        to_log = {}
        for idx, logit in enumerate(logits):
            if self._mask_decoder_gradient and idx == 1:
                assert logit.shape == mask.shape
                logit = ((1 - mask) * logit).view(x_hat.shape[0], -1).sum(dim=1)
                denom = (1 - mask).view(x_hat.shape[0], -1).sum(dim=1) + 1e-7
                logit = (logit / denom)
#            logit = logit.sum()
            grad = torch.autograd.grad(
                outputs=logit,
                inputs=x_hat,
                grad_outputs=torch.ones_like(logit),
                create_graph=True,
                only_inputs=True
            )[0]
            if self.mask_region_only:
                mask = batch["mask"]
                expected_shape = (real_data.shape[0], 1, *real_data.shape[2:])
                assert mask.shape == expected_shape, \
                    f"Expected shape: {expected_shape}. Got: {mask.shape}"
                grad = grad * (1 - mask)
            grad = grad.view(x_hat.shape[0], -1)
            grad_norm = self.norm(grad)
            gradient_pen = (grad_norm - 1)
            gradient_pen = self.clip(gradient_pen)
            to_backward += gradient_pen * self._lambd
            tag = "gradient_penalty"
            if idx > 0:
                tag = f"{tag}_{idx}"
            to_log[tag] = gradient_pen.mean().detach()
        x_hat.requires_grad = False
        return to_backward.view(-1), to_log


@CRITERION_REGISTRY.register_module
class EpsilonPenalty(GanCriterion):

    def __init__(self, weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight

    def d_loss(self, batch):
        real_scores = batch["real_scores"]
        real_scores = real_scores
        epsilon_penalty = 0
        for real in real_scores:
            epsilon_penalty += real.pow(2)
        to_log = dict(
            epsilon_penalty=epsilon_penalty.mean().detach()
        )
        return epsilon_penalty.view(-1), to_log


@CRITERION_REGISTRY.register_module
class PosePredictionPenalty(GanCriterion):

    def __init__(self, weight):
        self.weight = weight

    def d_loss(self, batch):
        real_pose_pred = batch["real_scores"][:, 1:]
        fake_pose_pred = batch["fake_scores"][self.fake_index][:, 1:]
        landmarks = batch["landmarks"].clone()
        # Normalize output to have a mean of 0
        landmarks = landmarks * 2 - 1
        real_pose_loss = (landmarks - real_pose_pred)**2
        fake_pose_loss = (landmarks - fake_pose_pred)**2
        to_log = dict(
            real_pose_loss=real_pose_loss.mean().detach(),
            fake_pose_loss=fake_pose_loss.mean().detach()
        )
        to_backward = ((real_pose_loss + fake_pose_loss) * 0.5)
        return to_backward.view(-1), to_log


@CRITERION_REGISTRY.register_module
class L1Loss(GanCriterion):

    REQUIRES_D_SCORE = False

    def __init__(self, weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight

    def g_loss(self, batch: dict):
        real = batch["img"]
        fake = batch["fake_data"][self.fake_index]
        mask = batch["mask"]
        l1_loss = torch.abs((real - fake) * (1 - mask)).view(real.shape[0], -1)
        denom = (1 - mask).view(real.shape[0], -1).sum(dim=1)
        l1_loss = l1_loss.sum(dim=1) / denom
        l1_loss = l1_loss * self.weight
        return l1_loss, dict(
            l1_loss=l1_loss.detach()
        )
