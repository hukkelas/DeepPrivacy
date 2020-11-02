from .build import CRITERION_REGISTRY
from .adversarial_loss import WGANCriterion, RGANCriterion, RaGANCriterion
from .loss import GradientPenalty, EpsilonPenalty, PosePredictionPenalty, L1Loss
from .optimizer import LossOptimizer