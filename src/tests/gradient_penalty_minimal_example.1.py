import torch
from torch import nn
from copy import deepcopy
from torch.optim import Adam
import numpy as np
from src.utils import to_cuda

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.set_printoptions(precision=10)


discriminator1 = nn.Sequential(
    nn.Conv2d(3, 16, (4,4)),
    nn.LeakyReLU(0.2),
    nn.Conv2d(16, 1, (1,1)),
    nn.LeakyReLU(0.2),
).cuda()

discriminator2 = nn.Sequential(
    nn.Conv2d(3, 16, (4,4)),
    nn.LeakyReLU(0.2),
    nn.Conv2d(16, 1, (1,1)),
    nn.LeakyReLU(0.2),
).cuda()
#discriminator1 = nn.Linear(2, 1, bias=True).cuda()
#discriminator2 = nn.Linear(2, 1, bias=True)
discriminator2.load_state_dict(deepcopy(discriminator1.state_dict()))

discriminator2 = nn.DataParallel(discriminator2).cuda()

optim1 = Adam(discriminator1.parameters())
optim2 = Adam(discriminator2.parameters())



def gradient_penalty(real_data, fake_data, discriminator):
    epsilon_shape = [real_data.shape[0]] + [1]*(real_data.dim() - 1)
    epsilon = torch.rand(epsilon_shape)
    epsilon = to_cuda(epsilon)
    epsilon = epsilon.to(fake_data.dtype)
    real_data = real_data.to(fake_data.dtype)
    x_hat = epsilon * real_data + (1-epsilon) * fake_data.detach()
    x_hat.requires_grad = True
    logits = discriminator(x_hat)
    logits = logits.sum() 
    grad = torch.autograd.grad(
        outputs=logits,
        inputs=x_hat,
        grad_outputs=torch.ones(logits.shape).to(fake_data.device),
        create_graph=True
    )[0] #.view(x_hat.shape[0], -1)
    grad = grad.view(x_hat.shape[0], -1)

    grad_penalty = ((grad.norm(p=2, dim=1) - 1)**2)
    return grad_penalty


x_real = torch.randn((16, 3,4,4)).cuda()
x_fake = torch.randn((16, 3,4,4)).cuda()

penalty = gradient_penalty(x_fake, x_real, discriminator1).sum()

optim1.zero_grad()
penalty.backward()
optim1.step()

y_out1 = discriminator1(x_real)


penalty = gradient_penalty(x_fake, x_real, discriminator2).sum()
#print(penalty)
optim2.zero_grad()
penalty.backward()
optim2.step()

y_out2 = discriminator2(x_real)

print(y_out1 - y_out2)

