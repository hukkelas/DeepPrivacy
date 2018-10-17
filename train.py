import tensorboardX
import torch
from torch.autograd import Variable
import torchvision
import utils
from utils import init_weights, load_checkpoint, save_checkpoint, to_cuda
from models import Generator, Discriminator
import tqdm
from torchsummary import summary
import os
from dataloaders import load_mnist
from options import load_options


def gradient_penalty(real_data, fake_data, discriminator):
    epsilon_shape = [real_data.shape[0]] + [1]*(real_data.dim() -1)
    epsilon = torch.rand(epsilon_shape)
    epsilon = to_cuda(epsilon)
    x_hat = epsilon * real_data + (1-epsilon) * fake_data
    x_hat = to_cuda(Variable(x_hat, requires_grad=True))

    logits = discriminator(x_hat)
    grad = torch.autograd.grad(
        outputs=logits,
        inputs=x_hat,
        grad_outputs=to_cuda(torch.ones(logits.shape)),
        create_graph=True
    )[0].view(x_hat.shape[0], -1)
    grad_penalty = ((grad.norm(p=2, dim=1) -1)**2).mean()
    return grad_penalty


def init_model():
    discriminator = Discriminator()
    generator = Generator()
    to_cuda([discriminator, generator])
    discriminator.apply(init_weights)
    generator.apply(init_weights)

    summary(discriminator, (1, 28, 28))
    summary(generator, (1, 100))
    return discriminator, generator


def generate_noise(batch_size, noise_dim):
    z = Variable(torch.randn(batch_size, noise_dim))
    z = to_cuda(z)
    return z


def save_images(writer, images, global_step, directory):
    filename = "step{}.jpg".format(global_step)
    filepath = os.path.join(directory, filename)
    torchvision.utils.save_image(images, filepath, nrow=10)
    image_grid = torchvision.utils.make_grid(images, normalize=True, nrow=10)
    writer.add_image("Image", image_grid, global_step)


def main(options):
    data_loader = load_mnist(options.batch_size)
    
    discriminator, generator = init_model()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=options.learning_rate, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=options.learning_rate, betas=(0.5, 0.999))

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


    """ run """
    writer = tensorboardX.SummaryWriter(options.summaries_dir)

    z_sample = generate_noise(100, options.noise_dim)
    for epoch in range(start_epoch, options.num_epochs):
        for i, (real_data, _) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
            generator.train()
            global_step = epoch * len(data_loader) + i + 1

            real_data = Variable(real_data)
            real_data = to_cuda(real_data)
            z = generate_noise(real_data.shape[0], options.noise_dim)

            fake_data = generator(z)

            # Train Discriminator
            real_logits = discriminator(real_data)
            fake_logits = discriminator(fake_data.detach())

            wasserstein_distance = real_logits.mean() - fake_logits.mean()  # Wasserstein-1 Distance
            gradient_pen = gradient_penalty(real_data.data, fake_data.data, discriminator)
            D_loss = -wasserstein_distance + gradient_pen * 10.0

            discriminator.zero_grad()
            D_loss.backward()
            d_optimizer.step()

            # Log data
            writer.add_scalar('discriminator/wasserstein-distance', wasserstein_distance.item(), global_step=global_step)
            writer.add_scalar('discriminator/gradient-penalty', gradient_pen.item(), global_step=global_step)

            if global_step % options.n_critic == 0:
                # Train Generator
                z = generate_noise(real_data.shape[0], options.noise_dim)
                fake_data = generator(z)
                fake_logits = discriminator(fake_data)
                G_loss = -fake_logits.mean()

                discriminator.zero_grad()
                generator.zero_grad()
                G_loss.backward()
                g_optimizer.step()

                writer.add_scalars('generator',
                                {"loss": G_loss.item()},
                                global_step=global_step)
                total_loss = D_loss + G_loss

                writer.add_scalars('total',
                                  {'loss': total_loss.item()},
                                  global_step=global_step)

            if (global_step+1) % 100 == 0:
                generator.eval()
                fake_data_sample = (generator(z_sample).data + 1) / 2.0
                save_images(writer, fake_data_sample, global_step, options.generated_data_dir)


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