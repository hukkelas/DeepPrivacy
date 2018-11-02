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


def generate_noise(batch_size, noise_dim, labels, label_size):
    z = Variable(torch.randn(batch_size, noise_dim))
    if label_size > 0:
        assert labels.shape[0] == batch_size, "Label size: {}, batch size: {}".format(labels.shape, batch_size)
        labels_onehot = torch.zeros((batch_size, label_size))
        idx = range(batch_size)
        labels_onehot[idx, labels.long().squeeze()] = 1
        assert labels_onehot.sum() == batch_size, "Was : {} expected:{}".format(labels_onehot.sum(), batch_size) 
        
        z[:, :label_size] = labels_onehot
    z = to_cuda(z)
    return z


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

def log_model_graphs(summaries_dir, generator, discriminator):

    with tensorboardX.SummaryWriter(summaries_dir + "_generator") as w:
        dummy_input = to_cuda(torch.zeros((1, generator.model.noise_dim, 1, 1)))
        w.add_graph(generator.model, input_to_model=dummy_input)
    with tensorboardX.SummaryWriter(summaries_dir + "_discriminator") as w:
        dummy_input = to_cuda(torch.zeros((1, discriminator.model.image_channels, discriminator.model.current_input_imsize, discriminator.model.current_input_imsize)))
        w.add_graph(discriminator.model, input_to_model=dummy_input)


def main(options):
    data_loader = load_dataset(options.dataset, options.batch_size, options.imsize)

    discriminator, generator = init_model(options.imsize, options.noise_dim, options.start_channel_size, options.image_channels, options.label_size)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=options.learning_rate, betas=(0.0, 0.999))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=options.learning_rate, betas=(0.0, 0.999))
    
    """ load checkpoint """

    try:
        ckpt = load_checkpoint(options.checkpoint_dir)
        start_epoch = ckpt['epoch']
        discriminator.load_state_dict(ckpt['discriminator'])
        generator.load_state_dict(ckpt['generator'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
    except:
        print(' [*] No checkpoint!')
        start_epoch = 0
    
    # Print Image stats
    real_data = next(iter(data_loader))[0]
    print("Input images:", real_data.max(), real_data.min())

    """ run """

    current_imsize = options.imsize
    current_channels = options.start_channel_size // 2
    writer = tensorboardX.SummaryWriter(options.summaries_dir)

    transition_variable = 1.
    transition_iters = 600000 // options.batch_size * options.batch_size
    transition_step = 0
    
    labels = torch.arange(0, options.label_size).repeat(100 // options.label_size)[:100].view(-1, 1) if options.label_size > 0 else None
    z_sample = generate_noise(100, options.noise_dim, labels, options.label_size)

    log_model_graphs(options.summaries_dir, generator, discriminator)
    is_transitioning = False
    global_step = 0
    start_time = time.time()
    label_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for epoch in range(start_epoch, options.num_epochs):
        for i, (real_data, labels) in tqdm.tqdm(enumerate(data_loader), 
                                           total=len(data_loader),
                                           desc="Global step: {:10.0f}".format(global_step)):
            batch_start_time = time.time()
            generator.train()
            labels = to_cuda(Variable(labels))
            global_step += 1 * options.batch_size
            if is_transitioning:
                transition_variable = ((global_step-1) % transition_iters) / transition_iters
            #print(real_data.min(), real_data.max(), real_data.mean())
            real_data = preprocess_images(real_data, transition_variable)
            #print(real_data.min(), real_data.max(), real_data.mean())
            z = generate_noise(real_data.shape[0], options.noise_dim, labels, options.label_size)

            # Forward G
            fake_data = generator(z, transition_variable)

            # Train Discriminator
            real_scores, real_logits = discriminator(real_data, transition_variable)
            fake_scores, fake_logits = discriminator(fake_data.detach(), transition_variable)
            wasserstein_distance = (real_scores - fake_scores).squeeze()  # Wasserstein-1 Distance
            gradient_pen = gradient_penalty(real_data.data, fake_data.data, discriminator, transition_variable)
            # Epsilon penalty
            epsilon_penalty = (real_scores ** 2).squeeze()

            # Label loss penalty
            label_penalty_discriminator = to_cuda(torch.Tensor([0]))
            if options.label_size > 0:
                label_penalty_reals = label_criterion(real_logits, labels)
                label_penalty_fakes = label_criterion(fake_logits, labels)
                label_penalty_discriminator = (label_penalty_reals + label_penalty_fakes).squeeze()
            assert wasserstein_distance.shape == gradient_pen.shape
            assert wasserstein_distance.shape == epsilon_penalty.shape
            D_loss = -wasserstein_distance + gradient_pen * 10 + epsilon_penalty * 0.001 + label_penalty_discriminator


            D_loss = D_loss.mean()
            discriminator.zero_grad()
            D_loss.backward()
            d_optimizer.step()


            # Forward G
            fake_scores, fake_logits = discriminator(fake_data, transition_variable)
            
            # Label loss penalty
            label_penalty_generator = label_criterion(fake_logits, labels).squeeze() if options.label_size > 0 else to_cuda(torch.Tensor([0]))
    
            
            G_loss = (-fake_scores + label_penalty_generator).mean()
            #G_loss = (-fake_scores).mean()
            discriminator.zero_grad()
            generator.zero_grad()
            G_loss.backward()
            g_optimizer.step()


            nsec_per_img = (time.time() - batch_start_time) / options.batch_size
            total_time = (time.time() - start_time) / 60
            # Log data
            writer.add_scalar('discriminator/wasserstein-distance', wasserstein_distance.mean().item(), global_step=global_step)
            writer.add_scalar('discriminator/gradient-penalty', gradient_pen.mean().item(), global_step=global_step)
            writer.add_scalar("discriminator/real-score", real_scores.mean().item(), global_step=global_step)
            writer.add_scalar("discriminator/fake-score", fake_scores.mean().item(), global_step=global_step)
            writer.add_scalar("discriminator/epsilon-penalty", epsilon_penalty.mean().item(), global_step=global_step)
            writer.add_scalar("stats/transition-value", transition_variable, global_step=global_step)
            writer.add_scalar("stats/nsec_per_img", nsec_per_img, global_step=global_step)
            writer.add_scalar("stats/training_time_minutes", total_time, global_step=global_step)
            writer.add_scalar("discriminator/label-penalty", label_penalty_discriminator.mean(), global_step=global_step)
            writer.add_scalar("generator/label-penalty", label_penalty_generator.mean(), global_step=global_step)


            if (global_step) % (options.batch_size*100) == 0:
                generator.eval()
                fake_data_sample = normalize_img(generator(z_sample, transition_variable).data)
                #print(fake_data_sample.max(), fake_data_sample.min())
                save_images(writer, fake_data_sample, global_step, options.generated_data_dir)

                # Save input images
                imsize = real_data.shape[2]
                filename = "reals{0}_{1}x{1}.jpg".format(global_step, imsize)
                filepath = os.path.join(options.generated_data_dir, filename)
                to_save = normalize_img(real_data[:100])
                torchvision.utils.save_image(to_save, filepath, nrow=10)
                
            
            if global_step % transition_iters == 0:
                #print("TRANSITION SWITCH")

                if global_step % (transition_iters*2) == 0:
                    # Stop transitioning
                    is_transitioning = False
                    transition_variable = 1.0
                elif current_imsize < options.max_imsize:
                    discriminator.extend(current_channels)
                    generator.extend(current_channels)
                    transition_step += 1
                    current_imsize *= 2
                    current_channels = current_channels // 2
                    discriminator.summary(), generator.summary()
                    data_loader = load_dataset(options.dataset, options.batch_size, current_imsize)
                    is_transitioning = True
                    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=options.learning_rate, betas=(0.0, 0.999), weight_decay=0)
                    g_optimizer = torch.optim.Adam(generator.parameters(), lr=options.learning_rate, betas=(0.0, 0.999), weight_decay=0)


                    fake_data_sample = normalize_img(generator(z_sample, transition_variable).data)
                    os.makedirs("lol", exist_ok=True)
                    filepath = os.path.join("lol", "test.jpg")
                    torchvision.utils.save_image(fake_data_sample[:100], filepath, nrow=10)
                    log_model_graphs(options.summaries_dir, generator, discriminator)


                    break



        filename = "Epoch_{}.ckpt".format(epoch)
        filepath = os.path.join(options.checkpoint_dir, filename)
        save_checkpoint({'epoch': epoch + 1,
                            'D': discriminator.state_dict(),
                            'G': generator.state_dict(),
                            'd_optimizer': d_optimizer.state_dict(),
                            'g_optimizer': g_optimizer.state_dict()},
                            filepath,
                            max_keep=2)

if __name__ == '__main__':
    options = load_options()
    main(options)
