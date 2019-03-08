import time
import os
import numpy as np
import tensorboardX
import torch
from apex.optimizers import FusedAdam
from apex import amp
import torchvision
import utils
from utils import load_checkpoint, save_checkpoint, to_cuda
from unet_model import Generator, Discriminator
from dataloaders_v2 import load_celeba_condition, load_ffhq_condition, load_yfcc100m
from options import load_options, print_options, DEFAULT_IMSIZE
from metrics import fid

torch.backends.cudnn.benchmark = True


def gradient_penalty(real_data, fake_data, discriminator, condition, landmarks):
    epsilon_shape = [real_data.shape[0]] + [1]*(real_data.dim() - 1)
    epsilon = torch.rand(epsilon_shape)
    epsilon = to_cuda(epsilon)
    epsilon = epsilon.to(fake_data.dtype)
    real_data = real_data.to(fake_data.dtype)
    x_hat = epsilon * real_data + (1-epsilon) * fake_data.detach()
    x_hat.requires_grad = True
    logits = discriminator(x_hat, condition, landmarks)
    grad = torch.autograd.grad(
        outputs=logits,
        inputs=x_hat,
        grad_outputs=to_cuda(torch.ones(logits.shape)).to(fake_data.dtype),
        create_graph=True
    )[0].view(x_hat.shape[0], -1)
    grad_penalty = ((grad.norm(p=2, dim=1) - 1)**2)
    return grad_penalty.to(fake_data.dtype)


class NetworkWrapper(torch.nn.Module):

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, *inputs):
        return self.network(*inputs)

    def extend(self, channel_size):
        self.network.extend(channel_size)

    def update_transition_value(self, value):
        self.network.transition_value = value

    def new_parameters(self):
        return self.network.new_parameters()


def init_model(imsize, pose_size, start_channel_dim, image_channels):
    discriminator = Discriminator(image_channels,
                                  imsize,
                                  start_channel_dim,
                                  pose_size)
    generator = Generator(pose_size, start_channel_dim, image_channels)
    to_cuda([discriminator, generator])
    discriminator = NetworkWrapper(discriminator)
    generator = NetworkWrapper(generator)

    return discriminator, generator


def save_images(writer, images, global_step, directory):
    imsize = images.shape[2]
    filename = "fakes{0}_{1}x{1}.jpg".format(global_step, imsize)
    filepath = os.path.join(directory, filename)
    torchvision.utils.save_image(images, filepath, nrow=10)
    image_grid = torchvision.utils.make_grid(images, nrow=10)
    writer.add_image("Image", image_grid, global_step)


def denormalize_img(image):
    image = (image+1)/2
    image = utils.clip(image, 0, 1)
    return image


def load_dataset(dataset, batch_size, imsize):
    if dataset == "celeba":
        return load_celeba_condition(batch_size, imsize)
    if dataset == "ffhq":
        return load_ffhq_condition(batch_size, imsize)
    if dataset == "yfcc100m":
        return load_yfcc100m(batch_size, imsize)


class DataPrefetcher():

    def __init__(self, loader, transition_variable):
        self.pool = torch.nn.AvgPool2d(2, 2)
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload(transition_variable)

    def preload(self, transition_variable):
        try:
            self.next_image, self.next_condition, self.next_landmark = next(self.loader)
        except StopIteration:
            self.next_image = None
            self.next_condition = None
            self.next_landmark = None
            return
        with torch.cuda.stream(self.stream):
            self.next_image = self.next_image.cuda(non_blocking=True).float()
            self.next_condition = self.next_condition.cuda(non_blocking=True).float()
            self.next_landmark = self.next_landmark.cuda(non_blocking=True)
            
            self.next_image = self.next_image / 255
            self.next_image = self.next_image*2 - 1

            self.next_condition = self.next_condition / 255
            self.next_condition = self.next_condition*2 - 1

            self.next_image = interpolate_image(self.pool,
                                                self.next_image,
                                                transition_variable)
            self.next_condition = interpolate_image(self.pool,
                                                    self.next_condition,
                                                    transition_variable)
    
    def next(self, transition_variable):
        torch.cuda.current_stream().wait_stream(self.stream)
        next_image = self.next_image
        next_condition = self.next_condition
        next_landmark = self.next_landmark
        self.preload(transition_variable)
        return next_image, next_condition, next_landmark


def interpolate_image(pool, images, transition_variable):
    y = pool(images)
    y = torch.nn.functional.interpolate(y, scale_factor=2)

    images = utils.get_transition_value(y, images, transition_variable)
    return images


class Trainer:

    def __init__(self, options):

        # Set Hyperparameters
        self.batch_size_schedule = options.batch_size
        self.batch_size = options.batch_size[options.imsize]
        self.dataset = options.dataset
        self.num_epochs = options.num_epochs
        self.learning_rate = options.learning_rate
        self.running_average_generator_decay = options.running_average_generator_decay
        self.pose_size = options.pose_size

        # Image settings
        self.current_imsize = options.imsize
        self.image_channels = 3
        self.max_imsize = options.max_imsize

        # Logging variables
        self.generated_data_dir = options.generated_data_dir
        self.checkpoint_dir = options.checkpoint_dir
        self.summaries_dir = options.summaries_dir

        # Transition settings
        self.transition_variable = 1.
        self.transition_iters = options.transition_iters
        self.is_transitioning = False
        self.transition_step = 0
        self.start_channel_size = options.start_channel_size
        self.latest_switch = 0
        current_channels = options.start_channel_size
        self.transition_channels = [
            current_channels,
            current_channels,
            current_channels,
            current_channels//2,
            current_channels//4,
            current_channels//8,
            current_channels//16,
            current_channels//32,
        ]
        self.start_time = time.time()
        self.writer = tensorboardX.SummaryWriter(options.summaries_dir)
        self.validation_writer = tensorboardX.SummaryWriter(
            os.path.join(options.summaries_dir, "validation"))
        if not self.load_checkpoint():
            self.discriminator, self.generator = init_model(options.imsize,
                                                            self.pose_size,
                                                            options.start_channel_size,
                                                            self.image_channels)
            self.init_running_average_generator()
            self.extend_models()
            self.init_optimizers()
        self.dataloader_train, self.dataloader_val = load_dataset(
            self.dataset, self.batch_size, self.current_imsize)

        self.log_variable("stats/batch_size", self.batch_size)
        self.discriminator.update_transition_value(self.transition_variable)
        self.generator.update_transition_value(self.transition_variable)

    def save_checkpoint(self, epoch):
        filename = "step_{}.ckpt".format(self.global_step)
        filepath = os.path.join(self.checkpoint_dir, filename)
        state_dict = {
            "epoch": epoch + 1,
            "D": self.discriminator.state_dict(),
            "G": self.generator.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            "batch_size": self.batch_size,
            "dataset": self.dataset,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "current_imsize": self.current_imsize,
            "max_imsize": self.max_imsize,
            "transition_variable": self.transition_variable,
            "transition_step": self.transition_step,
            "is_transitioning": self.is_transitioning,
            "start_channel_size": self.start_channel_size,
            "global_step": self.global_step,
            "image_channels": self.image_channels,
            "total_time": self.total_time,
            "batch_size_schedule": self.batch_size_schedule,
            "transition_iters":  self.transition_iters,
            "running_average_generator": self.running_average_generator.state_dict(),
            "running_average_generator_decay": self.running_average_generator_decay,
            "latest_switch": self.latest_switch,
            "pose_size": self.pose_size

        }
        save_checkpoint(state_dict,
                        filepath,
                        max_keep=2)

    def load_checkpoint(self):
        try:
            ckpt = load_checkpoint(self.checkpoint_dir)
            self.start_epoch = ckpt['epoch']
            print_options(ckpt)
            # Set Hyperparameters

            self.batch_size = ckpt["batch_size"]
            self.batch_size_schedule = ckpt["batch_size_schedule"]
            self.dataset = ckpt["dataset"]
            self.num_epochs = ckpt["num_epochs"]
            self.learning_rate = ckpt["learning_rate"]
            self.running_average_generator_decay = ckpt["running_average_generator_decay"]

            # Image settings
            self.current_imsize = ckpt["current_imsize"]
            self.image_channels = ckpt["image_channels"]
            self.max_imsize = ckpt["max_imsize"]
            self.pose_size = ckpt["pose_size"]

            # Logging variables
            # Transition settings
            self.transition_variable = ckpt["transition_variable"]
            self.transition_iters = ckpt["transition_iters"]
            self.is_transitioning = ckpt["is_transitioning"]
            self.transition_step = ckpt["transition_step"]
            self.latest_switch = ckpt["latest_switch"]
            self.global_step = ckpt["global_step"]
            self.start_time = time.time() - ckpt["total_time"] * 60

            current_channels = ckpt["start_channel_size"]
            self.transition_channels = [
                current_channels,
                current_channels,
                current_channels,
                current_channels//2,
                current_channels//4,
                current_channels//8,
                current_channels//16,
                current_channels//32,
            ]
            self.discriminator, self.generator = init_model(
                self.current_imsize // (2**self.transition_step),
                self.pose_size, current_channels,
                self.image_channels)
            self.init_running_average_generator()
            num_transitions = self.transition_step
            self.transition_step = 0
            self.current_imsize = DEFAULT_IMSIZE
            for i in range(num_transitions):
                self.extend_models()

            self.discriminator.load_state_dict(ckpt['D'])
            self.generator.load_state_dict(ckpt['G'])
            self.running_average_generator.load_state_dict(
                ckpt["running_average_generator"])
            self.init_optimizers()
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])

            return True
        except FileNotFoundError as e:
            print(e)
            print(' [*] No checkpoint!')
            self.start_epoch = 0
            self.global_step = 0
            return False

    def init_running_average_generator(self):
        self.running_average_generator = Generator(self.pose_size,
                                                   self.start_channel_size,
                                                   self.image_channels)
        self.running_average_generator = NetworkWrapper(
            self.running_average_generator)
        self.running_average_generator = to_cuda(
            self.running_average_generator)
        self.running_average_generator = self.running_average_generator.float()

    def extend_running_average_generator(self, current_channels):
        g = self.running_average_generator
        g.extend(current_channels)
        for avg_param, cur_param in zip(g.new_parameters(), self.generator.new_parameters()):
            assert avg_param.data.shape == cur_param.data.shape
            avg_param.data = cur_param.data
        self.running_average_generator = g.float()

    def extend_models(self):
        current_channels = self.transition_channels[self.transition_step]
        self.discriminator.extend(current_channels)
        self.generator.extend(current_channels)
        self.extend_running_average_generator(current_channels)

        self.current_imsize *= 2

        self.batch_size = self.batch_size_schedule[self.current_imsize]
        self.log_variable("stats/batch_size", self.batch_size)
        self.transition_step += 1

    def update_running_average_generator(self):
        for avg_parameter, current_parameter in zip(
                self.running_average_generator.parameters(),
                self.generator.parameters()):
            avg_parameter.data = self.running_average_generator_decay*avg_parameter + \
                ((1-self.running_average_generator_decay) * current_parameter.float())

    def init_optimizers(self):
        self.d_optimizer = FusedAdam(self.discriminator.parameters(),
                                     lr=self.learning_rate,
                                     betas=(0.0, 0.99))
        self.g_optimizer = FusedAdam(self.generator.parameters(),
                                     lr=self.learning_rate,
                                     betas=(0.0, 0.99))
        self.generator, self.g_optimizer = amp.initialize(self.generator,
                                                         self.g_optimizer,
                                                         keep_batchnorm_fp32=False,
                                                         opt_level='O1',
                                                         loss_scale="dynamic"
                                                         )
        self.discriminator, self.d_optimizer = amp.initialize(self.discriminator,
                                                             self.d_optimizer,
                                                             keep_batchnorm_fp32=False,
                                                             opt_level='O1',
                                                             loss_scale="dynamic"
                                                             )                                                         

    def log_variable(self, name, value, log_to_validation=False):
        if log_to_validation:
            self.validation_writer.add_scalar(name, value,
                                              global_step=self.global_step)
        else:
            self.writer.add_scalar(name, value, global_step=self.global_step)

    def save_transition_image(self, before):

        prefetcher = DataPrefetcher(self.dataloader_train, self.transition_variable)
        real_image, condition, landmark = prefetcher.next(self.transition_variable)

        fake_data = self.generator(condition, landmark)
        fake_data = denormalize_img(fake_data.detach())[:8]
        real_data = denormalize_img(real_image)[:8]
        condition = denormalize_img(condition)[:8]
        to_save = torch.cat((real_data, condition, fake_data))
        imname = "transition_after{}.jpg".format(self.current_imsize//2)
        if before:
            imname = "transition_before{}.jpg".format(self.current_imsize)
        os.makedirs("lol", exist_ok=True)
        filepath = os.path.join("lol",
                                imname)
        torchvision.utils.save_image(
            to_save, filepath, nrow=8
        )

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
        data_prefetcher = DataPrefetcher(self.dataloader_val,
                                         self.transition_variable)
        for idx in range(len(self.dataloader_val)):
            real_data, condition, landmarks = data_prefetcher.next(self.transition_variable)
            fake_data = self.running_average_generator(condition,
                                                       landmarks)
            real_score = self.discriminator(real_data, condition, landmarks)
            fake_score = self.discriminator(fake_data.detach(), condition,
                                            landmarks)
            wasserstein_distance = (real_score - fake_score).squeeze()
            epsilon_penalty = (real_score**2).squeeze()
            real_scores.append(real_score.mean().item())
            fake_scores.append(fake_score.mean().item())
            wasserstein_distances.append(wasserstein_distance.mean().item())
            epsilon_penalties.append(epsilon_penalty.mean().item())

            fake_data = denormalize_img(fake_data.detach())
            real_data = denormalize_img(real_data)

            start_idx = idx*self.batch_size
            end_idx = (idx+1)*self.batch_size
            real_images[start_idx:end_idx] = real_data.cpu().float()
            fake_images[start_idx:end_idx] = fake_data.cpu().float()
            del real_data, fake_data, real_score, fake_score
        if self.current_imsize >= 64:
            fid_val = fid.calculate_fid(real_images, fake_images, 8)
            print("FID:", fid_val)
            self.log_variable("stats/fid", np.mean(fid_val), True)
        self.log_variable('discriminator/wasserstein-distance',
                          np.mean(wasserstein_distances), True)
        self.log_variable("discriminator/real-score",
                          np.mean(real_scores), True)
        self.log_variable("discriminator/fake-score",
                          np.mean(fake_scores), True)
        self.log_variable("discriminator/epsilon-penalty",
                          np.mean(epsilon_penalties), True)
        directory = os.path.join(self.generated_data_dir, "validation")
        os.makedirs(directory, exist_ok=True)
        save_images(self.validation_writer, fake_images[:64], self.global_step,
                    directory)
        self.discriminator.train()

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            prefetcher = DataPrefetcher(self.dataloader_train, self.transition_variable)
            for i in range(len(self.dataloader_train)):
                batch_start_time = time.time()
                self.generator.train()
                if self.is_transitioning:
                    self.transition_variable = (
                        (self.global_step-1) % self.transition_iters) / self.transition_iters
                    self.discriminator.update_transition_value(
                        self.transition_variable)
                    self.generator.update_transition_value(
                        self.transition_variable)
                    self.running_average_generator.update_transition_value(
                        self.transition_variable)
                real_data, condition, landmarks = prefetcher.next(self.transition_variable)
                torchvision.utils.save_image(denormalize_img(real_data), "test.jpg", nrow=10)
                # Forward G
                fake_data = self.generator(condition, landmarks)
                # Train Discriminator
                real_scores = self.discriminator(
                    real_data, condition, landmarks)
                fake_scores = self.discriminator(
                    fake_data.detach(), condition, landmarks)

                # Wasserstein-1 Distance
                wasserstein_distance = (real_scores - fake_scores).squeeze()
                gradient_pen = gradient_penalty(
                    real_data.data, fake_data.detach(), self.discriminator,
                    condition, landmarks)
                # Epsilon penalty
                epsilon_penalty = (real_scores ** 2).squeeze()

                assert wasserstein_distance.shape == epsilon_penalty.shape
                D_loss = - wasserstein_distance
                D_loss += gradient_pen * 10 + epsilon_penalty * 0.001

                D_loss = D_loss.mean()
                self.d_optimizer.zero_grad()
                with amp.scale_loss(D_loss, self.d_optimizer) as scaled_loss:
                    scaled_loss.backward()
                self.d_optimizer.step()

                # Forward G
                fake_scores = self.discriminator(
                    fake_data, condition, landmarks)
                G_loss = (-fake_scores).mean()

                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()
                with amp.scale_loss(G_loss, self.g_optimizer) as scaled_loss:
                    scaled_loss.backward()
                self.g_optimizer.step()

                nsec_per_img = (
                    time.time() - batch_start_time) / self.batch_size
                self.total_time = (time.time() - self.start_time) / 60
                # Log data
                self.log_variable(
                    'discriminator/wasserstein-distance',
                    wasserstein_distance.mean().item())
                self.log_variable(
                    'discriminator/gradient-penalty',
                    gradient_pen.mean().item())
                self.log_variable("discriminator/real-score",
                                  real_scores.mean().item())
                self.log_variable("discriminator/fake-score",
                                  fake_scores.mean().item())
                self.log_variable("discriminator/epsilon-penalty",
                                  epsilon_penalty.mean().item())
                self.log_variable("stats/transition-value",
                                  self.transition_variable)
                self.log_variable("stats/nsec_per_img", nsec_per_img)
                self.log_variable(
                    "stats/training_time_minutes", self.total_time)
                self.update_running_average_generator()
                self.global_step += self.batch_size

                if (self.global_step) % (self.batch_size*500) == 0:
                    self.generator.eval()
                    fake_data_sample = denormalize_img(
                        self.generator(condition, landmarks).detach().data)
                    save_images(self.writer, fake_data_sample,
                                self.global_step, self.generated_data_dir)

                    # Save input images
                    imsize = real_data.shape[2]
                    filename = "reals{0}_{1}x{1}.jpg".format(
                        self.global_step, imsize)
                    filepath = os.path.join(self.generated_data_dir, filename)
                    to_save = denormalize_img(real_data)
                    torchvision.utils.save_image(to_save, filepath, nrow=10)

                    filename = "condition{0}_{1}x{1}.jpg".format(
                        self.global_step, imsize)
                    filepath = os.path.join(self.generated_data_dir, filename)
                    to_save = denormalize_img(condition[:, :3])
                    torchvision.utils.save_image(to_save, filepath, nrow=10)

                if self.global_step//self.batch_size*self.batch_size % (2e5//self.batch_size * self.batch_size) == 0:
                    self.save_checkpoint(epoch)
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
                        self.save_checkpoint(epoch)
                    elif self.current_imsize < self.max_imsize:
                        # Save image before transition
                        self.save_transition_image(True)
                        self.extend_models()
                        del self.dataloader_train, self.dataloader_val
                        self.dataloader_train, self.dataloader_val = load_dataset(
                            self.dataset, self.batch_size, self.current_imsize)
                        self.is_transitioning = True

                        self.init_optimizers()
                        self.transition_variable = 0
                        self.discriminator.update_transition_value(
                            self.transition_variable)
                        self.generator.update_transition_value(
                            self.transition_variable)
                        
                        # Save image after transition
                        self.save_transition_image(False)
                        self.save_checkpoint(epoch)

                        break
            self.save_checkpoint(epoch)


if __name__ == '__main__':
    options = load_options()

    trainer = Trainer(options)
    trainer.train()
