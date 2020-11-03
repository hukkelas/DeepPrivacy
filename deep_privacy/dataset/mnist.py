import torchvision
import numpy as np
from .build import DATASET_REGISTRY


@DATASET_REGISTRY.register_module
class MNISTDataset(torchvision.datasets.MNIST):

    def __init__(
            self,
            dirpath,
            imsize,
            transform,
            train,
            **kwargs):
        super().__init__(dirpath, train=train, download=True)
        self.transform = transform
        self.imsize = imsize

    def get_mask(self):
        mask = np.ones((self.imsize, self.imsize), dtype=np.bool)
        offset = self.imsize // 4
        mask[offset:-offset, offset:-offset] = 0
        return mask

    def __getitem__(self, index):
        im, _target = super().__getitem__(index)
        im = im.resize((self.imsize, self.imsize))
        im = np.array(im)[:, :, None]
        im = im.repeat(3, -1)
        mask = self.get_mask()
        if self.transform:
            im = self.transform(im)

        return {
            "img": im,
            "mask": mask,
            "landmarks": np.zeros((0)).astype(np.float32)
        }
