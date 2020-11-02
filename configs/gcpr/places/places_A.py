
_base_config_ = "../../places2.py"

adversarial_loss = "WGANCriterion"
discriminator_criterions = [
    dict(
        type=adversarial_loss,
        fake_index=-1
    ),
    dict(
        type="GradientPenalty",
        lambd=10,
        mask_region_only=False,
        norm="L2",
        distance="L2",
        lazy_reg_interval=16,
        mask_decoder_gradient=False,
        fake_index=-1
    ),
    dict(
        type="EpsilonPenalty",
        weight=0.001,
        fake_index=-1
    )
]
