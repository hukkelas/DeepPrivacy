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
from dataloaders import load_mnist, load_cifar10
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


def init_model(imsize, noise_dim, start_channel_dim, image_channels):
    discriminator = Discriminator(image_channels, imsize, start_channel_dim)
    generator = Generator(noise_dim, start_channel_dim, image_channels)
    to_cuda([discriminator, generator])
    discriminator.apply(init_weights)
    generator.apply(init_weights)
    discriminator.summary()
    generator.summary()
    return discriminator, generator


def generate_noise(batch_size, noise_dim):
    z = Variable(torch.randn(batch_size, noise_dim))
    z = to_cuda(z)
    return z


def save_images(writer, images, global_step, directory):
    imsize = images.shape[2]
    filename = "step{0}_{1}x{1}.jpg".format(global_step, imsize)
    filepath = os.path.join(directory, filename)
    torchvision.utils.save_image(images, filepath, nrow=10)
    image_grid = torchvision.utils.make_grid(images, normalize=True, nrow=10)
    writer.add_image("Image", image_grid, global_step)

def normalize_img(image):
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


def main(options):
    data_loader = load_dataset(options.dataset, options.batch_size, options.imsize)
    
    discriminator, generator = init_model(options.imsize, options.noise_dim, options.start_channel_size, options.image_channels)
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


    """ run """
    current_imsize = options.imsize
    current_channels = options.start_channel_size // 2
    writer = tensorboardX.SummaryWriter(options.summaries_dir)
    transition_variable = 1
    transition_iters = 10000
    transition_step = 0
    z_sample = generate_noise(100, options.noise_dim)
    is_transitioning = False
    global_step = 0
    for epoch in range(start_epoch, options.num_epochs):
        for i, (real_data, _) in tqdm.tqdm(enumerate(data_loader), 
                                           total=len(data_loader),
                                           desc="Global step: {:10.0f}".format(global_step)):

            generator.train()
            global_step += 1
            if is_transitioning:
                transition_variable = ((global_step-1) % transition_iters) / transition_iters
            real_data = Variable(real_data)
            real_data = to_cuda(real_data)
            z = generate_noise(real_data.shape[0], options.noise_dim)

            fake_data = generator(z, transition_variable)

            # Train Discriminator
            real_logits = discriminator(real_data, transition_variable)
            fake_logits = discriminator(fake_data.detach(), transition_variable)
            
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
                fake_data = generator(z, transition_variable)
                fake_logits = discriminator(fake_data, transition_variable)
                G_loss = -fake_logits.mean()
                writer.add_scalars("generator/fadein-constant", {
                    "fadin constant": transition_variable},
                    global_step=global_step
                )
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
                fake_data_sample = normalize_img(generator(z_sample, transition_variable).data)
                #print(fake_data_sample.max(), fake_data_sample.min())
                save_images(writer, fake_data_sample, global_step, options.generated_data_dir)

                # Save input images
                os.makedirs("input_images", exist_ok=True)
                imsize = real_data.shape[2]
                filename = "step{0}_{1}x{1}.jpg".format(global_step, imsize)
                filepath = os.path.join("input_images", filename)
                #writer.add_image("Image", image_grid, global_step)
                torchvision.utils.save_image(normalize_img(real_data[:100]), filepath, nrow=10)

            
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
                    #save_images(writer, fake_data_sample, global_step, options.generated_data_dir)


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