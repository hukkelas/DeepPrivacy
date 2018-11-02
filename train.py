import tensorboardX
import torch
from torch.autograd import Variable
import torchvision
import utils
from utils import init_weights, load_checkpoint, save_checkpoint, to_cuda
from models import Generator, Discriminator, get_transition_value
import tqdm
from torchsummary import summary
import os
from dataloaders import load_mnist, load_cifar10, load_pokemon, load_celeba
from options import load_options
import time
torch.backends.cudnn.benchmark=True


def gradient_penalty(real_data, fake_data, discriminator, transition_variable):
    epsilon_shape = [real_data.shape[0]] + [1]*(real_data.dim() -1)
    epsilon = torch.rand(epsilon_shape)
    epsilon = to_cuda(epsilon)
    x_hat = epsilon * real_data + (1-epsilon) * fake_data.detach()
    x_hat = to_cuda(Variable(x_hat, requires_grad=True))

    logits, _ = discriminator(x_hat, transition_variable)
    grad = torch.autograd.grad(
        outputs=logits,
        inputs=x_hat,
        grad_outputs=to_cuda(torch.ones(logits.shape)),
        create_graph=True
    )[0].view(x_hat.shape[0], -1)
    grad_penalty = ((grad.norm(p=2, dim=1) - 1)**2)
    return grad_penalty

class DataParallellWrapper(torch.nn.Module):
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.forward_block = torch.nn.DataParallel(self.model)

    def forward(self, x, transition_variable):
        return self.forward_block(x)
    
    def extend(self, channel_size):
        self.model.extend(channel_size)
        self.forward_block = torch.nn.DataParallel(self.model)
    
    def summary(self):
        self.model.summary()


def init_model(imsize, noise_dim, start_channel_dim, image_channels, label_size):
    discriminator = Discriminator(image_channels, imsize, start_channel_dim, label_size)
    discriminator = DataParallellWrapper(discriminator)
    generator = Generator(noise_dim, start_channel_dim, image_channels)
    generator = DataParallellWrapper(generator)
    to_cuda([discriminator, generator])
    #discriminator.apply(init_weights)
    #generator.apply(init_weights)
    discriminator.summary()
    generator.summary()
    return discriminator, generator





def adjust_dynamic_range(data):
    return data*2-1

def save_images(writer, images, global_step, directory):
    imsize = images.shape[2]
    filename = "fakes{0}_{1}x{1}.jpg".format(global_step, imsize)
    filepath = os.path.join(directory, filename)
    torchvision.utils.save_image(images, filepath, nrow=10)
    image_grid = torchvision.utils.make_grid(images, normalize=True, nrow=10)
    writer.add_image("Image", image_grid, global_step)

def normalize_img(image):
    return (image+1)/2
    if image.shape[1] == 3:
        image[:, 0, :, :] = image[:, 0, :, :] * 0.2023 + 0.4914
        image[:, 1, :, :] = image[:, 1, :, :] * 0.1994 + 0.4822
        image[:, 2, :, :] = image[:, 2, :, :] * 0.2010 + 0.4465
    else:
        image = image * 0.5 + 0.5
    
    return image


def load_dataset(dataset, batch_size, imsize):
    if dataset == "mnist":
        return load_mnist(batch_size, imsize)
    if dataset == "cifar10":
        return load_cifar10(batch_size, imsize)
    if dataset == "celeba":
        return load_celeba(batch_size, imsize)



def preprocess_images(images, transition_variable):
    images = Variable(images)
    images = to_cuda(images)
    images = adjust_dynamic_range(images)
    # Compute averaged image
    s = images.shape
    y = images.view([-1, s[1], s[2]//2, 2, s[3]//2, 2])
    y = y.mean(dim=3, keepdim=True).mean(dim=5, keepdim=True)
    y = y.repeat([1, 1, 1, 2, 1, 2])
    y = y.view(-1, s[1], s[2], s[3])
    
    images = get_transition_value(y, images, transition_variable)

    return images



class Trainer:

    def load_checkpoint(self, checkpoint_dir):
        try:
            ckpt = load_checkpoint(options.checkpoint_dir)
            self.start_epoch = ckpt['epoch']
            self.discriminator.load_state_dict(ckpt['discriminator'])
            self.generator.load_state_dict(ckpt['generator'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 0
        self.global_step = 0
    
    def generate_noise(self, batch_size, labels):
        z = Variable(torch.randn(batch_size, self.noise_dim))
        if self.label_size > 0:
            assert labels.shape[0] == batch_size, "Label size: {}, batch size: {}".format(labels.shape, batch_size)
            labels_onehot = torch.zeros((batch_size, self.label_size))
            idx = range(batch_size)
            labels_onehot[idx, labels.long().squeeze()] = 1
            assert labels_onehot.sum() == batch_size, "Was : {} expected:{}".format(labels_onehot.sum(), batch_size) 
            
            z[:, :self.label_size] = labels_onehot
        z = to_cuda(z)
        return z


    def log_model_graphs(self):
        with tensorboardX.SummaryWriter(self.summaries_dir + "_generator") as w:
            dummy_input = to_cuda(torch.zeros((1, self.generator.model.noise_dim, 1, 1)))
            w.add_graph(self.generator.model, input_to_model=dummy_input)
        with tensorboardX.SummaryWriter(self.summaries_dir + "_discriminator") as w:
            imsize= self.discriminator.model.current_input_imsize
            image_channels = self.discriminator.model.image_channels
            dummy_input = to_cuda(torch.zeros((1, 
                                               image_channels,
                                               imsize,
                                               imsize)))
            w.add_graph(self.discriminator.model, input_to_model=dummy_input)

    def init_optimizers(self):
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), 
                                            lr=self.learning_rate, betas=(0.0, 0.999))
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(),
                                            lr=self.learning_rate, betas=(0.0, 0.999))

    def log_variable(self, name, value):
        self.writer.add_scalar(name, value, global_step=self.global_step)

    def __init__(self, options):

        # Set Hyperparameters
        self.batch_size = options.batch_size
        self.dataset = options.dataset
        self.num_epochs = options.num_epochs
        self.label_size = options.label_size
        self.noise_dim = options.noise_dim
        self.learning_rate = options.learning_rate

        # Image settings
        self.current_imsize = options.imsize
        self.current_channels = options.start_channel_size
        self.max_imsize = options.max_imsize

        # Logging variables
        self.generated_data_dir = options.generated_data_dir
        self.checkpoint_dir = options.checkpoint_dir
        self.summaries_dir = options.summaries_dir
        
        # Transition settings
        self.transition_variable = 1.
        self.transition_iters = 6e5 // options.batch_size * options.batch_size
        self.is_transitioning = False
        self.transition_step = 0
        self.transition_channels = [
            self.current_channels,
            self.current_channels,
            self.current_channels,
            self.current_channels//2,
            self.current_channels//4,
            self.current_channels//8,
            self.current_channels//16,
            self.current_channels//32,                                                            
            
            ]

        self.start_time = time.time()


        self.data_loader = load_dataset(self.dataset, self.batch_size, options.imsize)

        self.discriminator, self.generator = init_model(options.imsize, options.noise_dim, options.start_channel_size, options.image_channels, options.label_size)
        self.init_optimizers()
        
        
        """ load checkpoint """
        self.load_checkpoint(self.checkpoint_dir)
        self.writer = tensorboardX.SummaryWriter(options.summaries_dir)

        
        labels = torch.arange(0, options.label_size).repeat(100 // options.label_size)[:100].view(-1, 1) if options.label_size > 0 else None
        self.z_sample = self.generate_noise(100, labels)

        self.log_model_graphs()
        
        

        self.label_criterion = torch.nn.CrossEntropyLoss(reduction='none')

    
    def train(self):

        for epoch in range(self.start_epoch, self.num_epochs):
            for i, (real_data, labels) in tqdm.tqdm(enumerate(self.data_loader), 
                                            total=len(self.data_loader),
                                            desc="Global step: {:10.0f}".format(self.global_step)):
                batch_start_time = time.time()
                self.generator.train()
                labels = to_cuda(Variable(labels))
                self.global_step += 1 * self.batch_size
                if self.is_transitioning:
                    self.transition_variable = ((self.global_step-1) % self.transition_iters) / self.transition_iters
                real_data = preprocess_images(real_data, self.transition_variable)
                z = self.generate_noise(real_data.shape[0], labels)

                # Forward G
                fake_data = self.generator(z, self.transition_variable)

                # Train Discriminator
                real_scores, real_logits = self.discriminator(real_data, self.transition_variable)
                fake_scores, fake_logits = self.discriminator(fake_data.detach(), self.transition_variable)

                wasserstein_distance = (real_scores - fake_scores).squeeze()  # Wasserstein-1 Distance
                gradient_pen = gradient_penalty(real_data.data, fake_data.data, self.discriminator, self.transition_variable)
                # Epsilon penalty
                epsilon_penalty = (real_scores ** 2).squeeze()

                # Label loss penalty
                label_penalty_discriminator = to_cuda(torch.Tensor([0]))
                if self.label_size > 0:
                    label_penalty_reals = self.label_criterion(real_logits, labels)
                    label_penalty_fakes = self.label_criterion(fake_logits, labels)
                    label_penalty_discriminator = (label_penalty_reals + label_penalty_fakes).squeeze()
                assert wasserstein_distance.shape == gradient_pen.shape
                assert wasserstein_distance.shape == epsilon_penalty.shape
                D_loss = -wasserstein_distance + gradient_pen * 10 + epsilon_penalty * 0.001 + label_penalty_discriminator


                D_loss = D_loss.mean()
                self.d_optimizer.zero_grad()
                D_loss.backward()
                self.d_optimizer.step()


                # Forward G
                fake_scores, fake_logits = self.discriminator(fake_data, self.transition_variable)
                
                # Label loss penalty
                label_penalty_generator = self.label_criterion(fake_logits, labels).squeeze() if self.label_size > 0 else to_cuda(torch.Tensor([0]))
        
                
                G_loss = (-fake_scores + label_penalty_generator).mean()
                #G_loss = (-fake_scores).mean()
                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()
                G_loss.backward()
                self.g_optimizer.step()


                nsec_per_img = (time.time() - batch_start_time) / self.batch_size
                total_time = (time.time() - self.start_time) / 60
                # Log data
                self.log_variable('discriminator/wasserstein-distance', wasserstein_distance.mean().item())
                self.log_variable('discriminator/gradient-penalty', gradient_pen.mean().item())
                self.log_variable("discriminator/real-score", real_scores.mean().item())
                self.log_variable("discriminator/fake-score", fake_scores.mean().item())
                self.log_variable("discriminator/epsilon-penalty", epsilon_penalty.mean().item())
                self.log_variable("stats/transition-value", self.transition_variable)
                self.log_variable("stats/nsec_per_img", nsec_per_img)
                self.log_variable("stats/training_time_minutes", total_time)
                self.log_variable("discriminator/label-penalty", label_penalty_discriminator.mean())
                self.log_variable("generator/label-penalty", label_penalty_generator.mean())


                if (self.global_step) % (self.batch_size*100) == 0:
                    self.generator.eval()
                    fake_data_sample = normalize_img(self.generator(self.z_sample, self.transition_variable).data)
                    save_images(self.writer, fake_data_sample, self.global_step, self.generated_data_dir)

                    # Save input images
                    imsize = real_data.shape[2]
                    filename = "reals{0}_{1}x{1}.jpg".format(self.global_step, imsize)
                    filepath = os.path.join(self.generated_data_dir, filename)
                    to_save = normalize_img(real_data[:100])
                    torchvision.utils.save_image(to_save, filepath, nrow=10)
                    
                
                if self.global_step % self.transition_iters == 0:

                    if self.global_step % (self.transition_iters*2) == 0:
                        # Stop transitioning
                        self.is_transitioning = False
                        self.transition_variable = 1.0
                    elif self.current_imsize < self.max_imsize:
                        self.current_channels = self.transition_channels[self.transition_step]
                        self.discriminator.extend(self.current_channels)
                        self.generator.extend(self.current_channels)
                        self.current_imsize *= 2

                        
                        self.discriminator.summary(), self.generator.summary()
                        self.data_loader = load_dataset(self.dataset, self.batch_size, self.current_imsize)
                        self.is_transitioning = True

                        self.init_optimizers()
                        self.transition_variable = 0
                        
                        fake_data_sample = normalize_img(self.generator(self.z_sample, self.transition_variable).data)
                        os.makedirs("lol", exist_ok=True)
                        filepath = os.path.join("lol", "test.jpg")
                        torchvision.utils.save_image(fake_data_sample[:100], filepath, nrow=10)
                        self.log_model_graphs()
                        self.transition_step += 1

                        break



            filename = "Epoch_{}.ckpt".format(epoch)
            filepath = os.path.join(self.checkpoint_dir, filename)
            save_checkpoint({'epoch': epoch + 1,
                                'D': self.discriminator.state_dict(),
                                'G': self.generator.state_dict(),
                                'd_optimizer': self.d_optimizer.state_dict(),
                                'g_optimizer': self.g_optimizer.state_dict()},
                                filepath,
                                max_keep=2)

if __name__ == '__main__':
    options = load_options()

    trainer = Trainer(options)
    trainer.train()
