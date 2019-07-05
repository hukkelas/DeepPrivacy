import time
import os
import numpy as np
import tensorboardX
import torch

from apex import amp
import torchvision
import utils
from utils import load_checkpoint, save_checkpoint, to_cuda, reduce_tensor, amp_state_has_overflow, is_distributed, gather_tensor
from unet_model import Generator, Discriminator, DeepDiscriminator
from dataloaders_v2 import load_dataset
from options import load_options, print_options, DEFAULT_IMSIZE
from metrics import fid
import tqdm
import apex
from data_tools.data_utils import DataPrefetcher

if False:
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=10)
else:
    torch.backends.cudnn.benchmark = True


def isnan(x):
    return x != x


def check_overflow(grad):
    cpu_sum = float(grad.float().sum())
    if cpu_sum == float("inf") or cpu_sum == -float("inf") or cpu_sum != cpu_sum:
        return True
    return False


@amp.float_function
def gradient_penalty(real_data, fake_data, discriminator, condition, landmarks, loss_scaler):
    epsilon_shape = [real_data.shape[0]] + [1]*(real_data.dim() - 1)
    epsilon = torch.rand(epsilon_shape)
    epsilon = to_cuda(epsilon)
    epsilon = epsilon.to(fake_data.dtype)
    real_data = real_data.to(fake_data.dtype)
    x_hat = epsilon * real_data + (1-epsilon) * fake_data.detach()
    x_hat.requires_grad = True
    logits = discriminator(x_hat, condition, landmarks)
    logits = logits.sum() * loss_scaler.loss_scale()
    grad = torch.autograd.grad(
        outputs=logits,
        inputs=x_hat,
        grad_outputs=torch.ones(logits.shape).to(fake_data.dtype).to(fake_data.device),
        create_graph=True
    )[0] #.view(x_hat.shape[0], -1)
    grad = grad.view(x_hat.shape[0], -1)
    if check_overflow(grad):
        print("Overflow in gradient penalty calculation.")
        loss_scaler._loss_scale /= 2
        print("Scaling down loss to:", loss_scaler._loss_scale)
        return None
    grad = grad / loss_scaler.loss_scale()

    grad_penalty = ((grad.norm(p=2, dim=1) - 1)**2)
    return grad_penalty.to(fake_data.dtype)


class NetworkWrapper(torch.nn.Module):

    def __init__(self, network, distributed, local_rank):
        super().__init__()
        self.network = network
        self.distributed = distributed
        #self.forward_block = torch.nn.DataParallel(
        #    self.network,
        #    device_ids=list(range(torch.cuda.device_count() ))
        #)
        self.forward_block = to_cuda(self.network)

    def forward(self, *inputs):
        return self.forward_block(*inputs)

    def extend(self, channel_size):
        del self.forward_block
        if type(self.network) == Discriminator:
            network = Discriminator(self.network.image_channels,
                                    4,
                                    self.network.orig_start_channel_dim,
                                    self.network.num_poses*2)
        elif type(self.network) == DeepDiscriminator:
            network = DeepDiscriminator(self.network.image_channels,
                                    4,
                                    self.network.orig_start_channel_dim,
                                    self.network.num_poses*2)
        else:
            network = Generator(self.network.num_poses*2,
                                self.network.orig_start_channel_dim,
                                self.network.image_channels)
        for c in self.network.extension_channels:
            network.extend(c)
        network.load_state_dict(self.network.state_dict())
        network.extend(channel_size)
        del self.network
        self.network = network
        self.forward_block = torch.nn.DataParallel(
            self.network,
            device_ids=list(range(torch.cuda.device_count()))
        )
        

    def update_transition_value(self, value):
        self.network.transition_value = value

    def new_parameters(self):
        return self.network.new_parameters()

    def state_dict(self):
        return self.network.state_dict()
    
    def load_state_dict(self, dict):
        self.network.load_state_dict(dict)


def init_model(imsize, pose_size, start_channel_dim, image_channels, distributed, discriminator_model):
    if discriminator_model == "deep":
        d = DeepDiscriminator
    else:
        assert discriminator_model == "normal"
        d = Discriminator

    discriminator = d(image_channels,
                                  imsize,
                                  start_channel_dim,
                                  pose_size)
    generator = Generator(pose_size, start_channel_dim, image_channels)
    #to_cuda([discriminator, generator])
    discriminator, generator = wrap_models([discriminator, generator], distributed, 0)
    return discriminator, generator


def wrap_models(models, distributed, local_rank):
    if isinstance(models, tuple) or isinstance(models, list):
        return [NetworkWrapper(x, distributed, local_rank) for x in models]
    return NetworkWrapper(models, distributed, local_rank)


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


def preprocess_images(image, transition_variable, pool=torch.nn.AvgPool2d(2, 2)):
    image = image * 2 - 1
    image = interpolate_image(pool, image, transition_variable)
    return image


class Trainer:

    def __init__(self, options):
        self.prev_step = 0
        # Set Hyperparameters
        self.batch_size_schedule = options.batch_size
        self.batch_size = options.batch_size[options.imsize] // options.world_size
        self.dataset = options.dataset
        self.num_epochs = options.num_epochs
        self.learning_rate = options.learning_rate
        self.running_average_generator_decay = options.running_average_generator_decay
        self.pose_size = options.pose_size
        self.discriminator_model = options.discriminator_model
        self.full_validation = options.full_validation

        # Image settings
        self.current_imsize = options.imsize
        self.image_channels = 3
        self.max_imsize = options.max_imsize

        # Logging variables
        self.generated_data_dir = options.generated_data_dir
        self.checkpoint_dir = options.checkpoint_dir
        self.summaries_dir = options.summaries_dir
        self.model_name = options.model_name

        # GPU SETTINGS
        self.distributed = options.distributed
        self.local_rank = options.local_rank
        self.world_size = options.world_size

        # validation checkpoint
        self.next_validation_checkpoint = 0
        self.validation_checkpoint_skip = 2e5

        # Transition settings
        self.transition_variable = 1.
        self.transition_iters = options.transition_iters
        self.is_transitioning = False
        self.transition_step = 0
        self.start_channel_size = options.start_channel_size
        self.latest_switch = 0
        self.opt_level = options.opt_level
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
        if self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(
                os.path.join(options.summaries_dir, "train")
            )
            self.validation_writer = tensorboardX.SummaryWriter(
                os.path.join(options.summaries_dir, "val"))
        if not self.load_checkpoint():
            self.discriminator, self.generator = init_model(options.imsize,
                                                            self.pose_size,
                                                            options.start_channel_size,
                                                            self.image_channels,
                                                            self.distributed,
                                                            self.discriminator_model)
            self.init_running_average_generator()
            self.extend_models()
            self.init_optimizers()
        self.dataloader_train, self.dataloader_val = load_dataset(
            self.dataset, self.batch_size, self.current_imsize, self.distributed, self.full_validation)

        self.log_variable("stats/batch_size", self.batch_size)
        #self.discriminator.update_transition_value(self.transition_variable)
        #self.generator.update_transition_value(self.transition_variable)

        if self.opt_level == "O0":
            self.wgan_gp_scaler = amp.scaler.LossScaler(1)
        else:
            self.wgan_gp_scaler = amp.scaler.LossScaler(2**14)

    def save_checkpoint(self, epoch, filepath=None):
        if self.local_rank != 0:
            return
        if filepath is None:
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
            "pose_size": self.pose_size,
            "opt_level": self.opt_level,
            "discriminator_model": self.discriminator_model,
            "next_validation_checkpoint": self.next_validation_checkpoint
        }
        save_checkpoint(state_dict,
                        filepath,
                        max_keep=2)

    def load_checkpoint(self):
        try:
            map_location = lambda storage, loc: storage.cuda(self.local_rank)#"cuda:{}".format(self.local_rank) if torch.cuda.is_available() else "cpu"
            ckpt = load_checkpoint(self.checkpoint_dir, map_location=map_location)
            self.start_epoch = ckpt['epoch']
            print_options(ckpt)
            # Set Hyperparameters

            self.batch_size = ckpt["batch_size"]
            self.batch_size_schedule = ckpt["batch_size_schedule"]
            self.dataset = ckpt["dataset"]
            self.num_epochs = ckpt["num_epochs"]
            self.learning_rate = ckpt["learning_rate"]
            self.running_average_generator_decay = ckpt["running_average_generator_decay"]
            self.start_channel_size = ckpt["start_channel_size"]
            self.discriminator_model = ckpt["discriminator_model"]
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
            self.opt_level = ckpt["opt_level"]
            self.next_validation_checkpoint = self.global_step#ckpt["next_validation_checkpoint"]#
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
                self.image_channels,
                self.distributed,
                self.discriminator_model)
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
        self.running_average_generator = to_cuda(self.running_average_generator)
        self.running_average_generator = amp.initialize(self.running_average_generator, None, opt_level=self.opt_level)
        self.running_average_generator = NetworkWrapper(
            self.running_average_generator, self.distributed, self.local_rank)

    def extend_running_average_generator(self, current_channels):
        g = self.running_average_generator
        g.extend(current_channels)
        
        for avg_param, cur_param in zip(g.new_parameters(), self.generator.new_parameters()):
            assert avg_param.data.shape == cur_param.data.shape, "AVG param: {}, cur_param: {}".format(avg_param.shape, cur_param.shape)
            avg_param.data = cur_param.data
        self.running_average_generator = self.running_average_generator.network
        self.running_average_generator.cuda()
        self.running_average_generator = amp.initialize(self.running_average_generator, None, opt_level=self.opt_level)
        self.running_average_generator = NetworkWrapper(self.running_average_generator, self.distributed, self.local_rank)

    def extend_models(self):
        current_channels = self.transition_channels[self.transition_step]
        self.discriminator.extend(current_channels)
        self.generator.extend(current_channels)
        self.extend_running_average_generator(current_channels)

        self.current_imsize *= 2

        self.batch_size = self.batch_size_schedule[self.current_imsize] // self.world_size
        
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

    def initialize_amp(self):
        
        if False:#isinstance(self.generator, NetworkWrapper):
            self.generator = to_cuda(self.generator.network)
        else:
            to_cuda(self.generator)
        if False:#isinstance(self.discriminator, NetworkWrapper):
            self.discriminator = to_cuda(self.discriminator.network)
        else:
            to_cuda(self.discriminator)
        [self.generator, self.discriminator], [self.g_optimizer, self.d_optimizer] = amp.initialize(
            [self.generator, self.discriminator],
            [self.g_optimizer, self.d_optimizer],
            opt_level=self.opt_level,
            num_losses=4)
        #self.discriminator, self.generator = wrap_models([self.discriminator, self.generator], self.distributed, self.local_rank)

    def log_variable(self, name, value, log_to_validation=False):
        if self.local_rank == 0:
            if log_to_validation:
                self.validation_writer.add_scalar(name, value,
                                                global_step=self.global_step)
            else:
                self.writer.add_scalar(name, value, global_step=self.global_step)

    def save_transition_image(self, before):
        prefetcher = DataPrefetcher(self.dataloader_val, self.transition_variable, self.pose_size)
        real_image, condition, landmark = prefetcher.next(self.transition_variable)

        fake_data = self.generator(condition, landmark)
        if self.local_rank == 0:
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
                                         self.transition_variable,
                                         self.pose_size)
        for idx in tqdm.trange(len(self.dataloader_val), desc="Validating model!"):
            real_data, condition, landmarks = data_prefetcher.next(self.transition_variable)
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
            fake_images[start_idx:end_idx] = fake_data.detach().cpu().float()
            del real_data, fake_data, real_score, fake_score, wasserstein_distance, epsilon_penalty
        to_nhwc = lambda x: np.stack((x[:,0], x[:, 1], x[:, 2]), axis=3)
        fid_name = "{}_{}_{}".format(self.dataset, self.full_validation, self.current_imsize)
        real_images = to_nhwc(real_images)
        fake_images2 = to_nhwc(fake_images)
        if self.current_imsize >= 64:
            fid_val = fid.calculate_fid(real_images, fake_images2, False, 8, fid_name)
            if self.distributed:
                fid_val = torch.tensor(fid_val, device="cuda:{}".format(self.local_rank))
                fid_val = reduce_tensor(fid_val, self.world_size).item()
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
        if self.local_rank == 0:
            save_images(self.validation_writer, fake_images[:64], self.global_step,
                    directory)
        self.discriminator.train()
        del data_prefetcher

    def log_loss_scales(self):
        if self.local_rank != 0:
            return
        for loss_idx, loss_scaler in enumerate(amp._amp_state.loss_scalers):
            self.log_variable("amp/loss_scale_{}".format(loss_idx), loss_scaler._loss_scale)
    
    def train_step(self, real_data, condition, landmarks):
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
            condition, landmarks, self.wgan_gp_scaler)
        if gradient_pen is None:
            return None
        # Epsilon penalty
        epsilon_penalty = (real_scores ** 2).squeeze()
        assert epsilon_penalty.shape == gradient_pen.shape
        assert wasserstein_distance.shape == epsilon_penalty.shape
        D_loss = - wasserstein_distance
        D_loss += gradient_pen * 10 + epsilon_penalty * 0.001

        D_loss = D_loss.mean()
        self.d_optimizer.zero_grad()

        to_backward1 = - wasserstein_distance.mean()
        with amp.scale_loss(to_backward1, self.d_optimizer, loss_id=0) as scaled_loss:
            scaled_loss.backward(retain_graph=True)
        # Hacky way to solve https://github.com/pytorch/pytorch/issues/13273
        to_backward2 = gradient_pen.mean() * 10 + 0 * wasserstein_distance.mean()
        with amp.scale_loss(to_backward2, self.d_optimizer, loss_id=1) as scaled_loss:
            scaled_loss.backward(retain_graph=True)
        to_backward3 = epsilon_penalty.mean() * 0.001
        with amp.scale_loss(to_backward3, self.d_optimizer, loss_id=2) as scaled_loss:
            scaled_loss.backward()

        self.d_optimizer.step()

        # Forward G
        fake_scores = self.discriminator(
            fake_data, condition, landmarks)
        G_loss = (-fake_scores).mean()

        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
        with amp.scale_loss(G_loss, self.g_optimizer, loss_id=3) as scaled_loss:
            scaled_loss.backward()
        self.g_optimizer.step()
        return wasserstein_distance.mean().detach(), gradient_pen.mean().detach(), real_scores.mean().detach(), fake_scores.mean().detach(), epsilon_penalty.mean().detach()

    def train(self):
        for epoch in range(self.start_epoch, int(1e16)):
            prefetcher = DataPrefetcher(self.dataloader_train,
                                        self.transition_variable,
                                        self.pose_size)
            for i in range(len(self.dataloader_train)):
                batch_start_time = time.time()
                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()
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
                # Forward G
                res = self.train_step(real_data,
                                      condition,
                                      landmarks)
                if res is None:
                    continue
                wasserstein_distance, gradient_pen, real_scores, fake_scores, epsilon_penalty = res
                nsec_per_img = (
                    time.time() - batch_start_time) / self.batch_size
                self.total_time = (time.time() - self.start_time) / 60
                # Log data
                self.update_running_average_generator()
                validation_checkpoint = 30 * 10**6
                if self.global_step >= validation_checkpoint and (self.global_step - self.batch_size) < validation_checkpoint:
                    print("Saving global checkpoint for validation")
                    dirname = os.path.join("validation_checkpoints/{}".format(self.model_name))
                    os.makedirs(dirname, exist_ok=True)
                    fpath = os.path.join(dirname, "step_{}.ckpt".format(self.global_step))
                    self.save_checkpoint(self.global_step, fpath)
                if i % 50 == 0:
                    if False:#self.distributed:
                        wasserstein_distance = reduce_tensor(wasserstein_distance.data,
                                                             self.world_size)

                        gradient_pen = reduce_tensor(gradient_pen.data, self.world_size)
                        real_scores = reduce_tensor(real_scores.data, self.world_size)
                        fake_scores = reduce_tensor(fake_scores.data, self.world_size)
                        epsilon_penalty = reduce_tensor(epsilon_penalty.data, self.world_size)
                    torch.cuda.synchronize()
                    self.log_loss_scales()
                    if self.local_rank == 0 and not amp_state_has_overflow():
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
                    
                        
                self.global_step += self.batch_size*self.world_size
                
                if (self.global_step // self.batch_size * self.batch_size) % (self.batch_size*500) == 0:
                    self.generator.eval()
                    fake_data_sample = denormalize_img(
                        self.generator(condition, landmarks).detach().data)
                    if self.local_rank == 0:
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
                each_step = 4e6 if self.current_imsize <= 32 else 2e5
                if self.global_step > self.next_validation_checkpoint:
                    self.save_checkpoint(epoch)
                    self.next_validation_checkpoint += self.validation_checkpoint_skip
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
                            self.dataset, self.batch_size, self.current_imsize, self.distributed, self.full_validation)
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
                        self.log_variable("stats/batch_size", self.batch_size)
                        self.log_variable("stats/learning_rate", self.learning_rate)

                        break
            self.save_checkpoint(epoch)

if __name__ == '__main__':
    options = load_options()
    trainer = Trainer(options)
    trainer.train()
