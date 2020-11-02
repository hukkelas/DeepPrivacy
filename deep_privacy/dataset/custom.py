import pathlib
from deep_privacy import logger
from .utils import read_image


class CustomDataset:

    def __init__(self,
                 dirpath,
                 imsize: int,
                 transform,
                 percentage: float):
        dirpath = pathlib.Path(dirpath)
        self.dirpath = dirpath
        self.transform = transform
        self._percentage = percentage
        self.imsize = imsize
        assert self.dirpath.is_dir(),\
            f"Did not find dataset at: {dirpath}"
        self.image_paths = self._load_impaths()
        self.filter_images()

        logger.info(
            f"Dataset loaded from: {dirpath}. Number of samples:{len(self)}, imsize={imsize}")

    def _load_impaths(self):
        image_dir = self.dirpath.joinpath("images", str(self.imsize))
        image_paths = list(image_dir.glob("*.png"))
        assert len(image_paths) > 0,\
            f"Did not find images in: {image_dir}"
        image_paths.sort(key=lambda x: int(x.stem))
        return image_paths

    def get_mask(self, idx):
        raise NotImplementedError

    def __len__(self):
        return len(self.image_paths)

    def get_image(self, index):
        impath = self.image_paths[index]
        im = read_image(impath)
        return im

    def get_item(self, index):
        image = self.get_image(index)
        masks = self.get_mask(index)
        return {
            "img": image,
            "mask": masks,
        }

    def __getitem__(self, index):
        batch = self.get_item(index)
        if self.transform is None:
            return batch
        return self.transform(batch)

    def filter_images(self) -> None:
        if 0 < self._percentage < 1:
            num_images = max(1, int(len(self.image_paths) * self._percentage))
            self.image_paths = self.image_paths[:num_images]
