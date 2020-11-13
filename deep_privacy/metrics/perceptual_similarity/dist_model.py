import numpy as np
import torch
import os
from .base_model import BaseModel

from . import networks_basic as networks


class DistModel(BaseModel):
    def name(self):
        return self.model_name

    def initialize(
            self, model='net-lin', net='alex', colorspace='Lab',
            pnet_rand=False, pnet_tune=False, model_path=None, use_gpu=True,
            printNet=False, spatial=False, is_train=False, lr=.0001, beta1=0.5,
            version='0.1', gpu_ids=[0]):
        '''
        INPUTS
            model - ['net-lin'] for linearly calibrated network
                    ['net'] for off-the-shelf network
                    ['L2'] for L2 distance in Lab colorspace
                    ['SSIM'] for ssim in RGB colorspace
            net - ['squeeze','alex','vgg']
            model_path - if None, will look in weights/[NET_NAME].pth
            colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
            use_gpu - bool - whether or not to use a GPU
            printNet - bool - whether or not to print network architecture out
            spatial - bool - whether to output an array containing varying distances across spatial dimensions
            spatial_shape - if given, output spatial shape. if None then spatial shape is determined automatically via spatial_factor (see below).
            spatial_factor - if given, specifies upsampling factor relative to the largest spatial extent of a convolutional layer. if None then resized to size of input images.
            spatial_order - spline order of filter for upsampling in spatial mode, by default 1 (bilinear).
            is_train - bool - [True] for training mode
            lr - float - initial learning rate
            beta1 - float - initial momentum term for adam
            version - 0.1 for latest, 0.0 was original (with a bug)
            gpu_ids - int array - [0] by default, gpus to use
        '''
        BaseModel.initialize(self, use_gpu=use_gpu, gpu_ids=gpu_ids)

        self.model = model
        self.net = net
        self.is_train = is_train
        self.spatial = spatial
        self.gpu_ids = gpu_ids
        self.model_name = '%s [%s]' % (model, net)

        if(self.model == 'net-lin'):  # pretrained net + linear layer
            self.net = networks.PNetLin(
                pnet_rand=pnet_rand,
                pnet_tune=pnet_tune,
                pnet_type=net,
                use_dropout=True,
                spatial=spatial,
                version=version,
                lpips=True)
            kw = {}
            if not use_gpu:
                kw['map_location'] = 'cpu'

            if(not is_train):
                #                print('Loading model from: %s'%model_path)
                state_dict = torch.hub.load_state_dict_from_url(
                    "http://folk.ntnu.no/haakohu/checkpoints/perceptual_similarity/alex.pth", **kw)
                self.net.load_state_dict(state_dict, strict=False)

        elif(self.model == 'net'):  # pretrained network
            self.net = networks.PNetLin(
                pnet_rand=pnet_rand, pnet_type=net, lpips=False)
        elif(self.model in ['L2', 'l2']):
            # not really a network, only for testing
            self.net = networks.L2(use_gpu=use_gpu, colorspace=colorspace)
            self.model_name = 'L2'
        elif(self.model in ['DSSIM', 'dssim', 'SSIM', 'ssim']):
            self.net = networks.DSSIM(use_gpu=use_gpu, colorspace=colorspace)
            self.model_name = 'SSIM'
        else:
            raise ValueError("Model [%s] not recognized." % self.model)

        self.parameters = list(self.net.parameters())

        if self.is_train:  # training mode
            # extra network on top to go from distances (d0,d1) => predicted
            # human judgment (h*)
            self.rankLoss = networks.BCERankingLoss()
            self.parameters += list(self.rankLoss.net.parameters())
            self.lr = lr
            self.old_lr = lr
            self.optimizer_net = torch.optim.Adam(
                self.parameters, lr=lr, betas=(beta1, 0.999))
        else:  # test mode
            self.net.eval()

        if(use_gpu):
            self.net.to(gpu_ids[0])
            self.net = torch.nn.DataParallel(self.net, device_ids=gpu_ids)
            if(self.is_train):
                self.rankLoss = self.rankLoss.to(
                    device=gpu_ids[0])  # just put this on GPU0

        if(printNet):
            print('---------- Networks initialized -------------')
            networks.print_network(self.net)
            print('-----------------------------------------------')

    def forward(self, in0, in1, retPerLayer=False):
        ''' Function computes the distance between image patches in0 and in1
        INPUTS
            in0, in1 - torch.Tensor object of shape Nx3xXxY - image patch scaled to [-1,1]
        OUTPUT
            computed distances between in0 and in1
        '''

        return self.net.forward(in0, in1, retPerLayer=retPerLayer)