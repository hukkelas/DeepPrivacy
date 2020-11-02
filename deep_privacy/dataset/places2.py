import os
import numpy as np
import pathlib
from .mask_util import generate_mask
from .custom import CustomDataset
from .build import DATASET_REGISTRY


@DATASET_REGISTRY.register_module
class Places2Dataset(CustomDataset):

    def __init__(self, *args, is_train: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_train = is_train

    def _load_impaths(self):
        relevant_suffixes = [".png", ".jpg", ".jpeg"]
        image_dir = self.dirpath
        image_paths = []
        for dirpath, dirnames, filenames in os.walk(image_dir):
            for filename in filenames:
                path = pathlib.Path(dirpath, filename)
                if path.suffix in relevant_suffixes:
                    assert path.is_file()
                    image_paths.append(path)
        # Name format of: Places365_test_00136999.jpg
        image_paths.sort(key=lambda x: int(x.stem.split("_")[-1]))
        return image_paths

    def get_image(self, *args, **kwargs):
        im = super().get_image(*args, **kwargs)
        if len(im.shape) == 2:
            im = im[:, :, None]
            im = np.repeat(im, 3, axis=-1)
        return im

    def get_mask(self, idx):
        return generate_mask(
            (self.imsize, self.imsize), fixed_mask=not self.is_train)
