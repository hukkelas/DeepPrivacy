import torch
import logging
import time
from deep_privacy import torch_utils, logger
from deep_privacy.metrics import metric_api
from .base import HookBase, HOOK_REGISTRY
from deep_privacy.inference import infer
try:
    from apex import amp
except ImportError:
    pass


@HOOK_REGISTRY.register_module
class ImageSaveHook(HookBase):

    def __init__(self, ims_per_save: int, n_diverse_samples: int):
        self.ims_per_save = ims_per_save
        self.next_save_point = self.ims_per_save
        self.before_images = None
        self._n_diverse_samples = n_diverse_samples

    def state_dict(self):
        return {
            "next_save_point": self.next_save_point,
            "before_images": self.before_images}

    def load_state_dict(self, state_dict: dict):
        self.next_save_point = state_dict["next_save_point"]
        self.before_images = state_dict["before_images"]

    def after_step(self):
        if self.global_step() >= self.next_save_point:
            self.next_save_point += self.ims_per_save
            self.save_fake_images(True)
            self.save_fake_images(False)

    def save_fake_images(self, validation: bool):
        g = self.trainer.generator
        if validation:
            g = self.trainer.RA_generator
        fake_data, real_data, condition = self.get_images(g)
        fake_data = fake_data[:64]
        logger.save_images(
            "fakes", fake_data, denormalize=True, nrow=8,
            log_to_validation=validation)
        logger.save_images(
            "reals", real_data[:64], denormalize=True, log_to_writer=False,
            nrow=8,
            log_to_validation=validation)
        condition = condition[:64]
        logger.save_images(
            "condition", condition, log_to_writer=False, denormalize=True,
            nrow=8,
            log_to_validation=validation)
        self.save_images_diverse()

    def get_images(self, g):
        g.eval()
        batch = next(iter(self.trainer.dataloader_val))
        z = g.generate_latent_variable(batch["img"]).zero_()
        with torch.no_grad():
            fake_data_sample = g(**batch,
                                 z=z)
        g.train()
        return fake_data_sample, batch["img"], batch["condition"]

    @torch.no_grad()
    def save_images_diverse(self):
        """
            Generates images with several latent variables
        """
        g = self.trainer.RA_generator
        g.eval()
        batch = next(iter(self.trainer.dataloader_val))
        batch = {k: v[:8] for k, v in batch.items()}
        fakes = [batch["condition"].cpu()]
        for i in range(self._n_diverse_samples):
            z = g.generate_latent_variable(batch["img"])
            fake = g(**batch, z=z)
            fakes.append(fake.cpu())
        fakes = torch.cat(fakes)
        logger.save_images(
            "diverse", fakes, log_to_validation=True, nrow=8, denormalize=True)
        g.train()

    def before_extend(self):
        transition_value = 1
        self.trainer.RA_generator.update_transition_value(
            transition_value
        )
        fake_data, real_data, condition = self.get_images(
            self.trainer.RA_generator
        )
        before_images = [
            torch_utils.denormalize_img(x[:8])
            for x in [real_data, fake_data, condition]
        ]
        before_images = torch.cat((before_images), dim=0)
        self.before_images = before_images.cpu()

    def after_extend(self):
        transition_value = 0
        self.trainer.RA_generator.update_transition_value(
            transition_value
        )
        fake_data, real_data, condition = self.get_images(
            self.trainer.RA_generator
        )

        after_images = [
            torch_utils.denormalize_img(x[:8])
            for x in [real_data, fake_data, condition]
        ]
        after_images = torch.cat((after_images), dim=0)
        after_images = torch.nn.functional.avg_pool2d(after_images, 2)
        after_images = after_images.cpu()
        assert after_images.shape == self.before_images.shape
        diff = self.before_images - after_images
        to_save = torch.cat(
            (self.before_images, after_images, diff), dim=2)
        imsize = after_images.shape[-1]
        imname = f"transition/from_{imsize}"
        logger.save_images(imname, to_save,
                           log_to_writer=True, nrow=8 * 3)
        self.before_images = None


@HOOK_REGISTRY.register_module
class MetricHook(HookBase):

    def __init__(
            self,
            ims_per_log: int,
            fid_batch_size: int,
            lpips_batch_size: int,
            min_imsize_to_calculate: int):
        self.next_check = ims_per_log
        self.num_ims_per_fid = ims_per_log
        self.lpips_batch_size = lpips_batch_size
        self.fid_batch_size = fid_batch_size
        self.min_imsize_to_calculate = min_imsize_to_calculate

    def state_dict(self):
        return {"next_check": self.next_check}

    def load_state_dict(self, state_dict: dict):
        self.next_check = state_dict["next_check"]

    def after_step(self):
        if self.global_step() >= self.next_check:
            self.next_check += self.num_ims_per_fid
            if self.current_imsize() >= self.min_imsize_to_calculate:
                self.calculate_fid()

    def calculate_fid(self):
        logger.info("Starting calculation of FID value")
        generator = self.trainer.RA_generator
        real_images, fake_images = infer.infer_images(
            self.trainer.dataloader_val, generator,
            truncation_level=0
        )
        """
        # Remove FID calculation as holy shit this is expensive.
        cfg = self.trainer.cfg
        identifier = f"{cfg.dataset_type}_{cfg.data_val.dataset.percentage}_{self.current_imsize()}"
        transition_value = self.trainer.RA_generator.transition_value
        fid_val = metric_api.fid(
            real_images, fake_images,
            batch_size=self.fid_batch_size)
        logger.log_variable("stats/fid", np.mean(fid_val),
                            log_level=logging.INFO)
        """

        l1 = metric_api.l1(real_images, fake_images)
        l2 = metric_api.l1(real_images, fake_images)
        psnr = metric_api.psnr(real_images, fake_images)
        lpips = metric_api.lpips(
            real_images, fake_images, self.lpips_batch_size)
        logger.log_variable("stats/l1", l1, log_level=logging.INFO)
        logger.log_variable("stats/l2", l2, log_level=logging.INFO)
        logger.log_variable("stats/psnr", psnr, log_level=logging.INFO)
        logger.log_variable("stats/lpips", lpips, log_level=logging.INFO)


@HOOK_REGISTRY.register_module
class StatsLogger(HookBase):

    def __init__(
            self,
            num_ims_per_log: int):
        self.num_ims_per_log = num_ims_per_log
        self.next_log_point = self.num_ims_per_log
        self.start_time = time.time()
        self.num_skipped_steps = 0

    def state_dict(self):
        return {
            "total_time": (time.time() - self.start_time),
            "num_skipped_steps": self.num_skipped_steps
        }

    def load_state_dict(self, state_dict: dict):
        self.start_time = time.time() - state_dict["total_time"]
        self.num_skipped_steps = state_dict["num_skipped_steps"]

    def before_train(self):
        self.batch_start_time = time.time()
        self.log_dictionary({"stats/batch_size": self.trainer.batch_size()})

    def log_dictionary(self, to_log: dict):
        logger.log_dictionary(to_log)

    def after_step(self):
        has_gradient_penalty = "loss/gradient_penalty" in self.to_log
        if has_gradient_penalty or self.global_step() >= self.next_log_point:
            self.log_stats()
            self.log_dictionary(self.to_log)
            self.log_loss_scales()
            self.next_log_point = self.global_step() + self.num_ims_per_log

    def log_stats(self):
        time_spent = time.time() - self.batch_start_time
        num_steps = self.global_step() - self.next_log_point + self.num_ims_per_log
        num_steps = max(num_steps, 1)
        nsec_per_img = time_spent / num_steps
        total_time = (time.time() - self.start_time) / 60
        to_log = {
            "stats/nsec_per_img": nsec_per_img,
            "stats/batch_size": self.trainer.batch_size(),
            "stats/training_time_minutes": total_time,
        }
        self.batch_start_time = time.time()
        self.log_dictionary(to_log)

    def log_loss_scales(self):
        to_log = {f'amp/loss_scale_{loss_idx}': loss_scaler._loss_scale
                  for loss_idx, loss_scaler in enumerate(amp._amp_state.loss_scalers)}
        to_log['amp/num_skipped_gradients'] = self.num_skipped_steps
        self.log_dictionary(to_log)
