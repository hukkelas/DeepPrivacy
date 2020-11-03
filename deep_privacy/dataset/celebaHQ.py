from .mask_util import generate_mask
from .custom import CustomDataset
from .build import DATASET_REGISTRY


@DATASET_REGISTRY.register_module
class CelebAHQDataset(CustomDataset):

    def __init__(self, *args, is_train, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_train = is_train

    def _load_impaths(self):
        image_dir = self.dirpath.joinpath(str(self.imsize))
        image_paths = list(image_dir.glob("*.png"))
        image_paths.sort(key=lambda x: int(x.stem))
        return image_paths

    def get_mask(self, idx):
        return generate_mask(
            (self.imsize, self.imsize), fixed_mask=not self.is_train)
