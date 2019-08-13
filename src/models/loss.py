import torch
from apex import amp
from src import torch_utils


def check_overflow(grad):
    cpu_sum = float(grad.float().sum())
    if cpu_sum == float("inf") or cpu_sum == -float("inf") or cpu_sum != cpu_sum:
        return True
    return False

@amp.float_function
def gradient_penalty(real_data, fake_data, discriminator, condition, landmarks, loss_scaler):
    epsilon_shape = [real_data.shape[0]] + [1]*(real_data.dim() - 1)
    epsilon = torch.rand(epsilon_shape)
    epsilon = epsilon.to(fake_data.device, fake_data.dtype)
    real_data = real_data.to(fake_data.dtype)
    x_hat = epsilon * real_data + (1-epsilon) * fake_data.detach()
    x_hat.requires_grad = True
    logits = discriminator(x_hat, condition, landmarks)
    logits = logits.sum()
    grad = torch.autograd.grad(
        outputs=logits,
        inputs=x_hat,
        grad_outputs=torch.ones(logits.shape).to(fake_data.dtype).to(fake_data.device),
        create_graph=True
    )[0] 
    grad = grad.view(x_hat.shape[0], -1)

    grad_penalty = ((grad.norm(p=2, dim=1) - 1)**2)
    return grad_penalty.to(fake_data.dtype)


class WGANLoss:

    def __init__(self, discriminator, generator, opt_level):
        self.generator = generator
        self.discriminator = discriminator
        if opt_level == "O0":
            self.wgan_gp_scaler = amp.scaler.LossScaler(1)
        else:
            self.wgan_gp_scaler = amp.scaler.LossScaler(2**14)

    
    def update_optimizers(self, d_optimizer, g_optimizer):
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
    
    def compute_gradient_penalty(self, real_data, fake_data, condition, landmarks):
        epsilon_shape = [real_data.shape[0]] + [1]*(real_data.dim() - 1)
        epsilon = torch.rand(epsilon_shape)
        epsilon = epsilon.to(fake_data.device, fake_data.dtype)
        real_data = real_data.to(fake_data.dtype)
        x_hat = epsilon * real_data + (1-epsilon) * fake_data.detach()
        x_hat.requires_grad = True
        logits = self.discriminator(x_hat, condition, landmarks)
        logits = logits.sum()
        grad = torch.autograd.grad(
            outputs=logits,
            inputs=x_hat,
            grad_outputs=torch.ones(logits.shape).to(fake_data.dtype).to(fake_data.device),
            create_graph=True
        )[0] 
        grad = grad.view(x_hat.shape[0], -1)
        gradient_pen = ((grad.norm(p=2, dim=1) - 1)**2)
        to_backward = gradient_pen.sum() * 10 
        with amp.scale_loss(to_backward, self.d_optimizer, loss_id=1) as scaled_loss:
            scaled_loss.backward(retain_graph=True)
        return gradient_pen.detach().mean()
        
        
    def step(self, real_data, condition, landmarks):
        with torch.no_grad():
            fake_data = self.generator(condition, landmarks)
        # Train Discriminator
        real_scores = self.discriminator(
            real_data, condition, landmarks)
        fake_scores = self.discriminator(
            fake_data.detach(), condition, landmarks)
        # Wasserstein-1 Distance
        wasserstein_distance = (real_scores - fake_scores).squeeze()
        
        # Epsilon penalty
        epsilon_penalty = (real_scores ** 2).squeeze()

        self.d_optimizer.zero_grad()
        gradient_pen = self.compute_gradient_penalty(real_data, fake_data, condition, landmarks)

        to_backward1 = (- wasserstein_distance).sum()
        with amp.scale_loss(to_backward1, self.d_optimizer, loss_id=0) as scaled_loss:
            scaled_loss.backward(retain_graph=True)

        to_backward3 = epsilon_penalty.sum() * 0.001
        with amp.scale_loss(to_backward3, self.d_optimizer, loss_id=2) as scaled_loss:
            scaled_loss.backward()
        self.d_optimizer.step()
        if not torch_utils.finiteCheck(self.discriminator.parameters()):
            return None
        fake_data = self.generator(condition, landmarks)
        # Forward G
        for p in self.discriminator.parameters():
            p.requires_grad = False
        fake_scores = self.discriminator(
            fake_data, condition, landmarks)
        G_loss = (-fake_scores).sum()

        
        self.g_optimizer.zero_grad()
        with amp.scale_loss(G_loss, self.g_optimizer, loss_id=3) as scaled_loss:
            scaled_loss.backward()
        self.g_optimizer.step()
        for p in self.discriminator.parameters():
            p.requires_grad = True
        if not torch_utils.finiteCheck(self.generator.parameters()):
            return None

        return wasserstein_distance.mean().detach(), gradient_pen.mean().detach(), real_scores.mean().detach(), fake_scores.mean().detach(), epsilon_penalty.mean().detach()
        

    



