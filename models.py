
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import to_cuda, init_weights


class EqualizedConv2D(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size, padding):
        super(EqualizedConv2D, self).__init__()

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, padding=padding)
        self.conv.apply(init_weights)
        self.conv.bias.data = self.conv.bias.data.zero_() 
        self.conv = nn.utils.weight_norm(self.conv)
    
    def forward(self, x):
        return self.conv(x)

def conv_bn_relu(in_dim, out_dim, kernel_size, padding=0):
    return nn.Sequential(
        EqualizedConv2D(in_dim, out_dim, kernel_size, padding),
        PixelwiseNormalization(),
        nn.LeakyReLU()
    )


def get_transition_value(x_old, x_new, transition_variable):
    return (1-transition_variable) * x_old + transition_variable*x_new

class PixelwiseNormalization(nn.Module):

    def __init__(self):
        super(PixelwiseNormalization, self).__init__()

    def forward(self, x):
        factor = (x**2).mean(dim=1, keepdim=True)**0.5
        return x / factor

class UpSamplingBlock(nn.Module):
    def __init__(self):
        super(UpSamplingBlock, self).__init__()
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2)

class Generator(nn.Module):

    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        # Transition blockss
        self.new_end_model = EqualizedConv2D(128, 1, 1, 0)
        self.old_end_model = EqualizedConv2D(128, 1, 1, 0)
        self.new_block = nn.Sequential(
        )
        self.core_model = nn.Sequential(
            conv_bn_relu(noise_dim, 128, 3, 1),            
        )
        self.first_block = nn.Sequential(
            nn.Linear(noise_dim, noise_dim*4*4)
        )
    
    def extend(self, output_dim):
        # Find input shape
        input_dim = self.new_end_model.conv.weight.shape[1]
        self.core_model = nn.Sequential(
            self.core_model,
            self.new_block,
            UpSamplingBlock()
        )
        self.new_block = nn.Sequential(
            conv_bn_relu(input_dim, output_dim, 3, 1),
            conv_bn_relu(output_dim, output_dim, 3, 1)
        )
        self.new_block = to_cuda(self.new_block) 
        self.old_end_model = self.new_end_model
        self.new_end_model = EqualizedConv2D(output_dim, 1, 1,0)
        self.new_end_model = to_cuda(self.new_end_model)


    # x: Bx1x1x512
    def forward(self, x, transition_variable=1):
        x = x.view((x.shape[0], -1))
        x = self.first_block(x)
        x = x.view(x.shape[0], -1, 4, 4)
        
        x = self.core_model(x)
        x_old = self.old_end_model(x)
        x_new = self.new_block(x)
        x_new = self.new_end_model(x_new)        
        x = get_transition_value(x_old, x_new, transition_variable)
        return x


def conv_module(dim_in, dim_out, kernel_size, padding, image_width):
    return nn.Sequential(
        EqualizedConv2D(dim_in, dim_out, kernel_size, padding),
        nn.LeakyReLU()
    )

class Discriminator(nn.Module):

    def __init__(self, in_channels, imsize):
        super(Discriminator, self).__init__()
        self.image_channels = in_channels
        self.current_input_imsize = 4
        self.new_start_model = nn.Sequential(
            conv_module(in_channels,128,1,0,self.current_input_imsize)
        ) 
        self.old_start_model = nn.Sequential(
            conv_module(in_channels,128,1,0,self.current_input_imsize),
        ) 
        self.new_block = nn.Sequential()
        self.core_model = nn.Sequential(
            conv_module(128, 128, 3, 1, imsize),
            conv_module(128, 128, 4, 0, 1),            
        )
        self.linear = nn.Sequential(
            nn.Linear(128, 1),
        )
        
    def extend(self, input_dim):
        
        self.current_input_imsize *= 2
        output_dim = list(self.new_start_model.parameters())[1].shape[0]
        new_core_model = nn.Sequential()
        idx = 0
        if self.new_block is not None:
            for _, module in self.new_block.named_children():
                new_core_model.add_module(str(idx), module)
                idx +=1
        for _, module in self.core_model.named_children():
            new_core_model.add_module(str(idx), module)
            idx += 1
        self.core_model = new_core_model
        self.old_start_model = nn.Sequential(
            nn.AvgPool2d([2,2]),
        )
        for idx, (name, module) in enumerate(self.new_start_model.named_children()):
            self.old_start_model.add_module(str(idx+1), module)
        self.new_start_model = nn.Sequential(
            conv_module(self.image_channels, input_dim, 1,0,self.current_input_imsize)
        )
        self.new_start_model = to_cuda(self.new_start_model)
        self.new_block = nn.Sequential(
            conv_module(input_dim, input_dim, 3, 1, self.current_input_imsize),
            conv_module(input_dim, output_dim, 3, 1, self.current_input_imsize),
            nn.AvgPool2d([2, 2])
        )
        self.new_block = to_cuda(self.new_block)



    # x: Bx1x1x512
    def forward(self, x, transition_variable=1):
        x_old = self.old_start_model(x)
        x_new = self.new_start_model(x)
        x_new = self.new_block(x_new)
        x = get_transition_value(x_old, x_new, transition_variable)
        x = self.core_model(x)
        x = x.view(-1, 128)
        x = self.linear(x)
        return x


if __name__ == "__main__":

    # Test that logits is the same

    # real data
    z = to_cuda(torch.zeros((64, 128, 1,1)))
    d = Discriminator(1, 4).cuda()
    g = Generator(128).cuda()
    data = g(z, 1)
    logits = d(data, 1)


    print(logits)

    d.extend(32)
    g.extend(32)

    data = g(z, 0)
    lg = d(data, 0)
    print(lg)

