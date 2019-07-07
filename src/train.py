import time
import os
import numpy as np
import torch
from apex import amp
import torchvision
from src import utils
from src.utils import load_checkpoint, save_checkpoint, to_cuda, amp_state_has_overflow, wrap_models
from src.models.generator import Generator
from src.models.unet_model import init_model
from src.data_tools.dataloaders_v2 import load_dataset
from src import config_parser
from src.metrics import fid
from src.models import loss
import tqdm
import apex
from src.data_tools.data_utils import denormalize_img
from src import logger 

if False:
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=10)
else:
    torch.backends.cudnn.benchmark = True

class Trainer:

    def __init__(self, config):
        # Set Hyperparameters
        self.batch_size_schedule = config.train_config.batch_size_schedule
        self.dataset = config.dataset
        self.learning_rate = config.train_config.learning_rate
        self.running_average_generator_decay = config.models.generator.running_average_decay
        self.pose_size = config.models.pose_size
        self.discriminator_model = config.models.discriminator.structure
        self.full_validation = config.use_full_validation
        self.load_fraction_of_dataset = config.load_fraction_of_dataset

        # Image settings
        self.current_imsize = 4
        self.image_channels = 3
        self.max_imsize = config.max_imsize

        # Logging variables
        self.checkpoint_dir = config.checkpoint_dir
        self.model_name = self.checkpoint_dir.split("/")[-2]
        self.global_step = 0

        # Transition settings
        self.transition_variable = 1.
        self.transition_iters = config.train_config.transition_iters
        self.is_transitioning = False
        self.transition_step = 0
        self.start_channel_size = config.models.start_channel_size
        self.latest_switch = 0
        self.opt_level = config.train_config.amp_opt_level
        self.start_time = time.time()
        self.discriminator, self.generator = init_model(self.pose_size,
                                                            config.models.start_channel_size,
                                                            self.image_channels,
                                                            self.discriminator_model)
        self.init_running_average_generator()
        self.criterion = loss.WGANLoss(self.discriminator, self.generator, self.opt_level)
        if not self.load_checkpoint():
            self.extend_models()
            self.init_optimizers()
        
        self.logger = logger.Logger(config.summaries_dir, config.generated_data_dir)

        self.batch_size = self.batch_size_schedule[self.current_imsize]
        self.logger.log_variable("stats/batch_size", self.batch_size)


        
      
        self.num_ims_per_log = config.logging.num_ims_per_log
        self.next_log_point = self.global_step
        self.num_ims_per_save_image = config.logging.num_ims_per_save_image
        self.next_image_save_point = self.global_step 
        self.num_ims_per_checkpoint = config.logging.num_ims_per_checkpoint
        self.next_validation_checkpoint = self.global_step

        self.dataloader_train, self.dataloader_val = load_dataset(
            self.dataset, self.batch_size, self.current_imsize, self.full_validation, self.pose_size, self.load_fraction_of_dataset)
        

    def save_checkpoint(self, filepath=None):
        if filepath is None:
            filename = "step_{}.ckpt".format(self.global_step)
            filepath = os.path.join(self.checkpoint_dir, filename)
        state_dict = {
            "D": self.discriminator.state_dict(),
            "G": self.generator.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            "transition_step": self.transition_step,
            "is_transitioning": self.is_transitioning,
            "global_step": self.global_step,
            "total_time": self.total_time,
            "running_average_generator": self.running_average_generator.state_dict(),
            "latest_switch": self.latest_switch,
            "current_imsize": self.current_imsize,
            "transition_step": self.transition_step
        }
        save_checkpoint(state_dict,
                        filepath,
                        max_keep=2)

    def load_checkpoint(self):
        try:
            map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
            ckpt = load_checkpoint(self.checkpoint_dir, map_location=map_location)
            # Transition settings
            self.is_transitioning = ckpt["is_transitioning"]
            self.transition_step = ckpt["transition_step"]
            self.current_imsize = ckpt["current_imsize"]
            self.latest_switch = ckpt["latest_switch"]
            
            # Tracking stats
            self.global_step = ckpt["global_step"]
            self.start_time = time.time() - ckpt["total_time"] * 60
            
            # Models
            self.discriminator.load_state_dict(ckpt['D'])

            self.generator.load_state_dict(ckpt['G'])
            self.running_average_generator.load_state_dict(
                ckpt["running_average_generator"])
            to_cuda([self.generator, self.discriminator, self.running_average_generator])
            self.running_average_generator = amp.initialize(self.running_average_generator,
                                                            None, opt_level=self.opt_level)
            self.init_optimizers()
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
            return True
        except FileNotFoundError as e:
            print(e)
            print(' [*] No checkpoint!')
            return False

    def init_running_average_generator(self):
        self.running_average_generator = Generator(self.pose_size,
                                                   self.start_channel_size,
                                                   self.image_channels)
        self.running_average_generator = wrap_models(self.running_average_generator)
        to_cuda(self.running_average_generator)
        self.running_average_generator = amp.initialize(self.running_average_generator,
                                                        None, opt_level=self.opt_level)
        

    def extend_running_average_generator(self):
        g = self.running_average_generator
        g.extend()
        
        for avg_param, cur_param in zip(g.new_parameters(), self.generator.new_parameters()):
            assert avg_param.data.shape == cur_param.data.shape, "AVG param: {}, cur_param: {}".format(avg_param.shape, cur_param.shape)
            avg_param.data = cur_param.data
        to_cuda(g)
        self.running_average_generator = amp.initialize(self.running_average_generator, None, opt_level=self.opt_level)
        

    def extend_models(self):
        self.discriminator.extend()
        self.generator.extend()
        self.extend_running_average_generator()

        self.current_imsize *= 2

        self.batch_size = self.batch_size_schedule[self.current_imsize] 
        self.transition_step += 1

    def update_running_average_generator(self):
        for avg_parameter, current_parameter in zip(
                self.running_average_generator.parameters(),
                self.generator.parameters()):
            
            avg_parameter.data = self.running_average_generator_decay*avg_parameter + \
                ((1-self.running_average_generator_decay) * current_parameter.float())

    def init_optimizers(self):
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                     lr=self.learning_rate,
                                     betas=(0.0, 0.99))
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(),
                                     lr=self.learning_rate,
                                     betas=(0.0, 0.99))
        self.initialize_amp()
        self.criterion.update_optimizers(self.d_optimizer, self.g_optimizer)

    def initialize_amp(self):
        to_cuda([self.generator, self.discriminator])
        [self.generator, self.discriminator], [self.g_optimizer, self.d_optimizer] = amp.initialize(
            [self.generator, self.discriminator],
            [self.g_optimizer, self.d_optimizer],
            opt_level=self.opt_level,
            num_losses=4)

    def save_transition_image(self, before):
        self.dataloader_val.update_next_transition_variable(self.transition_variable)
        real_image, condition, landmark = next(iter(self.dataloader_val))

        fake_data = self.generator(condition, landmark)
        fake_data = denormalize_img(fake_data.detach())[:8]
        real_data = denormalize_img(real_image)[:8]
        condition = denormalize_img(condition)[:8]
        to_save = torch.cat((real_data, condition, fake_data))
        tag = "before" if before else "after"
        imsize = self.current_imsize if before else self.current_imsize // 2
        imname = "transition/{}_{}".format(tag, imsize)
        self.logger.save_images(imname, to_save, log_to_writer=False)

    def validate_model(self):
        real_scores = []
        fake_scores = []
        wasserstein_distances = []
        epsilon_penalties = []
        self.running_average_generator.eval()
        self.discriminator.eval()
        real_images = torch.zeros((len(self.dataloader_val)*self.batch_size,
                                   3,
                                   self.current_imsize,
                                   self.current_imsize))
        fake_images = torch.zeros((len(self.dataloader_val)*self.batch_size,
                                   3,
                                   self.current_imsize,
                                   self.current_imsize))
        with torch.no_grad():
            self.dataloader_val.update_next_transition_variable(self.transition_variable)
            for idx, (real_data, condition, landmarks) in enumerate(tqdm.tqdm(self.dataloader_val, desc="Validating model!")):
                fake_data = self.running_average_generator(condition,
                                                           landmarks)
                real_score = self.discriminator(real_data, condition, landmarks)
                fake_score = self.discriminator(fake_data.detach(), condition,
                                                landmarks)
                wasserstein_distance = (real_score - fake_score).squeeze()
                epsilon_penalty = (real_score**2).squeeze()
                real_scores.append(real_score.mean().detach().item())
                fake_scores.append(fake_score.mean().detach().item())
                wasserstein_distances.append(wasserstein_distance.mean().detach().item())
                epsilon_penalties.append(epsilon_penalty.mean().detach().item())

                fake_data = denormalize_img(fake_data.detach())
                real_data = denormalize_img(real_data)

                start_idx = idx*self.batch_size
                end_idx = (idx+1)*self.batch_size
                real_images[start_idx:end_idx] = real_data.cpu().float()
                fake_images[start_idx:end_idx] = fake_data.cpu().float()
                del real_data, fake_data, real_score, fake_score, wasserstein_distance, epsilon_penalty
        to_nhwc = lambda x: np.stack((x[:,0], x[:, 1], x[:, 2]), axis=3)
        fid_name = "{}_{}_{}".format(self.dataset, self.full_validation, self.current_imsize)
        real_images = to_nhwc(real_images)
        fake_images2 = to_nhwc(fake_images)
        if self.current_imsize >= 64:
            fid_val = fid.calculate_fid(real_images, fake_images2, False, 8, fid_name)
            self.logger.log_variable("stats/fid", np.mean(fid_val), True)
        self.logger.log_variable('discriminator/wasserstein-distance',
                          np.mean(wasserstein_distances), True)
        self.logger.log_variable("discriminator/real-score",
                          np.mean(real_scores), True)
        self.logger.log_variable("discriminator/fake-score",
                          np.mean(fake_scores), True)
        self.logger.log_variable("discriminator/epsilon-penalty",
                          np.mean(epsilon_penalties), True)
        self.logger.save_images("fakes", fake_images[:64], log_to_validation=True)
        self.discriminator.train()
        self.generator.train()

    def log_loss_scales(self):
        for loss_idx, loss_scaler in enumerate(amp._amp_state.loss_scalers):
            self.logger.log_variable("amp/loss_scale_{}".format(loss_idx), loss_scaler._loss_scale)
    
    def train_step(self, real_data, condition, landmarks):
        fake_data = self.generator(condition, landmarks)
        res = self.criterion.step(real_data, fake_data, condition, landmarks)
        if res is None:
            return
        wasserstein_distance, gradient_pen, real_scores, fake_scores, epsilon_penalty = res
        self.total_time = (time.time() - self.start_time) / 60
        if self.global_step >= self.next_log_point and not amp_state_has_overflow():
            time_spent = time.time() - batch_start_time
            nsec_per_img = time_spent / (self.global_step - self.next_log_point + self.num_ims_per_log) 
            self.logger.log_variable("stats/nsec_per_img", nsec_per_img)
            self.next_log_point = self.global_step + self.num_ims_per_log
            batch_start_time = time.time()
            self.log_loss_scales()
            self.logger.log_variable(
                'discriminator/wasserstein-distance',
                wasserstein_distance.item())
            self.logger.log_variable(
                'discriminator/gradient-penalty',
                gradient_pen.item())
            self.logger.log_variable("discriminator/real-score",
                            real_scores.item())
            self.logger.log_variable("discriminator/fake-score",
                            fake_scores.mean().item())
            self.logger.log_variable("discriminator/epsilon-penalty",
                            epsilon_penalty.item())
            self.logger.log_variable("stats/transition-value",
                            self.transition_variable)
            self.logger.log_variable("stats/batch_size", self.batch_size)
            self.logger.log_variable("stats/learning_rate", self.learning_rate)                        
            self.logger.log_variable(
                "stats/training_time_minutes", self.total_time)
            
    def update_transition_value(self):
        self.transition_variable = utils.compute_transition_value(
            self.global_step, self.is_transitioning, self.transition_iters
        )
        self.discriminator.update_transition_value(self.transition_variable)
        self.generator.update_transition_value(self.transition_variable)
        self.running_average_generator.update_transition_value(self.transition_variable)

    def maybe_save_validation_checkpoint(self):
        validation_checkpoint = 30 * 10**6
        if self.global_step >= validation_checkpoint and (self.global_step - self.batch_size) < validation_checkpoint:
            print("Saving global checkpoint for validation")
            dirname = os.path.join("validation_checkpoints/{}".format(self.model_name))
            os.makedirs(dirname, exist_ok=True)
            fpath = os.path.join(dirname, "step_{}.ckpt".format(self.global_step))
            self.save_checkpoint(self.global_step, fpath)

    def train(self):
        batch_start_time = time.time()
        while True:
            self.update_transition_value()
            self.dataloader_train.update_next_transition_variable(self.transition_variable)
            train_iter = iter(self.dataloader_train)
            next_transition_value = utils.compute_transition_value(
                self.global_step + self.batch_size, self.is_transitioning, self.transition_iters
            )
            self.dataloader_train.update_next_transition_variable(next_transition_value)
            for real_data, condition, landmarks in train_iter:
                self.logger.update_global_step(self.global_step)
                self.update_transition_value()
                self.train_step(real_data, condition, landmarks)
                
                # Log data
                self.update_running_average_generator()
                self.maybe_save_validation_checkpoint()    
                
                if self.global_step >= self.next_image_save_point:
                    self.next_image_save_point = self.global_step + self.num_ims_per_save_image
                    self.generator.eval()
                    with torch.no_grad():
                        fake_data_sample = denormalize_img(
                            self.generator(condition, landmarks).data)
                    self.logger.save_images("fakes", fake_data_sample[:64])
                    # Save input images
                    to_save = denormalize_img(real_data)
                    self.logger.save_images("reals", to_save[:64], log_to_writer=False)
                    to_save = denormalize_img(condition[:64, :3])
                    self.logger.save_images("condition", to_save, log_to_writer=False)
                if self.global_step > self.next_validation_checkpoint:
                    self.save_checkpoint()
                    self.next_validation_checkpoint += self.num_ims_per_checkpoint
                    self.validate_model()
                if self.global_step >= (self.latest_switch + self.transition_iters):
                    self.latest_switch += self.transition_iters
                    if self.is_transitioning:
                        # Stop transitioning
                        self.is_transitioning = False
                        self.transition_variable = 1.0
                        self.discriminator.update_transition_value(
                            self.transition_variable)
                        self.generator.update_transition_value(
                            self.transition_variable)
                        self.save_checkpoint()
                    elif self.current_imsize < self.max_imsize:
                        # Save image before transition
                        self.save_checkpoint()
                        self.save_transition_image(True)
                        self.extend_models()
                        del self.dataloader_train, self.dataloader_val
                        self.dataloader_train, self.dataloader_val = load_dataset(
                            self.dataset, self.batch_size, self.current_imsize, self.full_validation, self.pose_size, self.load_fraction_of_dataset)
                        self.is_transitioning = True

                        self.init_optimizers()
                        self.transition_variable = 0
                        self.discriminator.update_transition_value(
                            self.transition_variable)
                        self.generator.update_transition_value(
                            self.transition_variable)
                        
                        # Save image after transition
                        self.save_transition_image(False)
                        break
                self.global_step += self.batch_size
                next_transition_value = utils.compute_transition_value(
                    self.global_step + self.batch_size, self.is_transitioning, self.transition_iters
                )
                self.dataloader_train.update_next_transition_variable(next_transition_value)

if __name__ == '__main__':
    config = config_parser.initialize_and_validate_config()
    trainer = Trainer(config)
    trainer.train()
