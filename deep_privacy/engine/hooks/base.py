from deep_privacy.utils import Registry, build_from_cfg

HOOK_REGISTRY = Registry("HOOKS")


def build_hooks(cfg, trainer):
    for _hook in cfg.trainer.hooks:
        if _hook.type == "CheckpointHook":
            hook = build_from_cfg(
                _hook, HOOK_REGISTRY, output_dir=cfg.output_dir)
        else:
            hook = build_from_cfg(_hook, HOOK_REGISTRY)
        trainer.register_hook(_hook.type, hook)


class HookBase:

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass

    def after_extend(self):
        """
            Will be called after we increase resolution / model size
        """
        pass

    def before_extend(self):
        """
            Will be called before we increase resolution / model size
        """
        pass

    def load_state_dict(self, state_dict: dict):
        pass

    def state_dict(self):
        return None

    def global_step(self):
        return self.trainer.global_step

    def current_imsize(self):
        return self.trainer.current_imsize()

    def get_transition_value(self):
        return self.trainer.hooks["progressive_trainer_hook"].transition_value
