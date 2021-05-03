
import numpy as np
try:
    from skimage.measure import compare_ssim
except ImportError:
    from skimage.metrics import structural_similarity as compare_ssim
import torch
from torch.autograd import Variable

from . import dist_model


class PerceptualLoss(torch.nn.Module):
    # VGG using our perceptually-learned weights (LPIPS metric)
    def __init__(self, model='net-lin', net='alex', colorspace='rgb',
                 spatial=False, use_gpu=True, gpu_ids=[0]):
        # def __init__(self, model='net', net='vgg', use_gpu=True): # "default"
        # way of using VGG as a perceptual loss
        super(PerceptualLoss, self).__init__()
#        print('Setting up Perceptual loss...')
        self.use_gpu = use_gpu
        self.spatial = spatial
        self.gpu_ids = gpu_ids
        self.model = dist_model.DistModel()
        self.model.initialize(
            model=model,
            net=net,
            use_gpu=use_gpu,
            colorspace=colorspace,
            spatial=self.spatial,
            gpu_ids=gpu_ids)
#        print('...[%s] initialized'%self.model.name())
#        print('...Done')

    def forward(self, pred, target, normalize=False):
        """
        Pred and target are Variables.
        If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
        If normalize is False, assumes the images are already between [-1,+1]

        Inputs pred and target are Nx3xHxW
        Output pytorch Variable N long
        """

        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1

        return self.model.forward(target, pred)


def l2(p0, p1, range=255.):
    return .5 * np.mean((p0 / range - p1 / range)**2)


def psnr(p0, p1, peak=255.):
    return 10 * np.log10(peak**2 / np.mean((1. * p0 - 1. * p1)**2))


def dssim(p0, p1, range=255.):
    return (1 - compare_ssim(p0, p1, data_range=range, multichannel=True)) / 2.


def rgb2lab(in_img, mean_cent=False):
    from skimage import color
    img_lab = color.rgb2lab(in_img)
    if(mean_cent):
        img_lab[:, :, 0] = img_lab[:, :, 0] - 50
    return img_lab


def tensor2np(tensor_obj):
    # change dimension of a tensor object into a numpy array
    return tensor_obj[0].cpu().float().numpy().transpose((1, 2, 0))


def np2tensor(np_obj):
    # change dimenion of np array into tensor array
    return torch.Tensor(np_obj[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


def tensor2tensorlab(image_tensor, to_norm=True, mc_only=False):
    # image tensor to lab tensor
    from skimage import color

    img = tensor2im(image_tensor)
    img_lab = color.rgb2lab(img)
    if(mc_only):
        img_lab[:, :, 0] = img_lab[:, :, 0] - 50
    if(to_norm and not mc_only):
        img_lab[:, :, 0] = img_lab[:, :, 0] - 50
        img_lab = img_lab / 100.

    return np2tensor(img_lab)


def tensorlab2tensor(lab_tensor, return_inbnd=False):
    from skimage import color
    import warnings
    warnings.filterwarnings("ignore")

    lab = tensor2np(lab_tensor) * 100.
    lab[:, :, 0] = lab[:, :, 0] + 50

    rgb_back = 255. * np.clip(color.lab2rgb(lab.astype('float')), 0, 1)
    if(return_inbnd):
        # convert back to lab, see if we match
        lab_back = color.rgb2lab(rgb_back.astype('uint8'))
        mask = 1. * np.isclose(lab_back, lab, atol=2.)
        mask = np2tensor(np.prod(mask, axis=2)[:, :, np.newaxis])
        return (im2tensor(rgb_back), mask)
    else:
        return im2tensor(rgb_back)


def rgb2lab(input):
    from skimage import color
    return color.rgb2lab(input / 255.)
