import torch
import numpy as np
import argparse
import sys
sys.path.append("/workspace")
print(sys.path)
import os
from apex import amp
from train import NetworkWrapper, init_model, gradient_penalty
from utils import gather_tensor
from torch.optim import Adam
parser = argparse.ArgumentParser()

parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--overwrite_results", default=False, action="store_true")
options = parser.parse_args()
print(options)
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.set_printoptions(precision=10)


savepath = os.path.join("tests/.results/distributed_test_result.torch")
os.makedirs(os.path.dirname(savepath), exist_ok=True)
discriminator, generator = init_model(
    4,
    14,
    256,
    3,
    False,
    "normal"
)
discriminator = torch.nn.DataParallel(discriminator)
generator = torch.nn.DataParallel(generator)
fake_data = torch.randn((64, 3, 4, 4)).cuda()
x_in = torch.randn((64, 3, 4, 4)).cuda()
z_in = torch.randn((64, 14)).cuda()
x_out = generator(x_in, z_in)
wgan_gp_scaler = amp.scaler.LossScaler(1)

optimizer = Adam(discriminator.parameters())
penalty = gradient_penalty(x_in, x_out, discriminator, fake_data, z_in, wgan_gp_scaler).mean()
optimizer.zero_grad()
penalty.backward()
optimizer.step()

y_out = discriminator(x_in, fake_data, z_in)

if options.overwrite_results:
    print("Saving output")
    torch.save(y_out, savepath)
prev = torch.load(savepath, map_location=f"cuda:{options.local_rank}")
if not options.overwrite_results:
    print(abs(prev - y_out).sum())


"""
ds = [p.grad for p in discriminator.parameters()]
if options.overwrite_results:
    print("Saving grads")
    torch.save(grads, savepath)
#print(grads)
prev_grads = torch.load(savepath, map_location=f"cuda:{options.local_rank}")
assert len(prev_grads) == len(grads)
for grad1, grad2 in zip(grads, prev_grads):
    if grad1 is None:
        assert grad2 is None
        continue
    try:
        print((grad1 - grad2).mean())
    except TypeError as e:
        print(type(e))
        print(grad1, grad2)
        print(e)
        print("FUFUFUFU")
"""