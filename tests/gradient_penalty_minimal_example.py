import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import os

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.set_printoptions(precision=10)


discriminator1 = nn.Linear(1, 1, bias=True).cuda()

discriminator2 = nn.Linear(1, 1, bias=True)
discriminator2.load_state_dict(discriminator1.state_dict())
discriminator2 = nn.DataParallel(discriminator2).cuda()

optim1 = Adam(discriminator1.parameters())
optim2 = Adam(discriminator2.parameters())



# Define input
x_fake = torch.randn((1, 1)).cuda()

# Discriminator 1
real_data = x_fake.clone()
real_data.requires_grad=True
logits = discriminator1(real_data).sum()
grad = torch.autograd.grad(
    outputs=logits,
    inputs=real_data,
    grad_outputs=torch.ones(logits.shape).cuda(),
    create_graph=True
)[0]
penalty = grad.sum()
optim1.zero_grad()
penalty.backward()
optim1.step()
y_out1 = discriminator1(x_fake)

# Discriminator 2
real_data = x_fake.clone()
real_data.requires_grad=True
logits = discriminator2(real_data).sum()
grad = torch.autograd.grad(
    outputs=logits,
    inputs=real_data,
    grad_outputs=torch.ones(logits.shape).cuda(),
    create_graph=True
)[0]
penalty = grad.sum()
optim2.zero_grad()
penalty.backward()
optim2.step()
y_out2 = discriminator2(x_fake)

print(y_out1 - y_out2)