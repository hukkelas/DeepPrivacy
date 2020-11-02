import pathlib
import numpy as np
import torch
from .build import DATASET_REGISTRY
from .custom import CustomDataset


def load_torch(filepath: pathlib.Path):
    assert filepath.is_file(),\
        f"Did not find file. Looked at: {filepath}"
    return torch.load(filepath)


@DATASET_REGISTRY.register_module
class FDFDataset(CustomDataset):

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.load_landmarks()
        self.load_bounding_box()

    def load_bounding_box(self):
        filepath = self.dirpath.joinpath(
            "bounding_box", f"{self.imsize}.torch")
        bbox = load_torch(filepath)
        self.bounding_boxes = bbox[:len(self)]
        assert len(self.bounding_boxes) == len(self)

    def load_landmarks(self):
        filepath = self.dirpath.joinpath("landmarks.npy")
        assert filepath.is_file(), \
            f"Did not find landmarks at: {filepath}"
        landmarks = np.load(filepath).reshape(-1, 7, 2)
        landmarks = landmarks.astype(np.float32)
        self.landmarks = landmarks[:len(self)]
        assert len(self.landmarks) == len(self),\
            f"Number of images: {len(self)}, landmarks: {len(landmarks)}"

    def get_mask(self, idx):
        mask = np.ones((self.imsize, self.imsize), dtype=np.bool)
        bounding_box = self.bounding_boxes[idx]
        x0, y0, x1, y1 = bounding_box
        mask[y0:y1, x0:x1] = 0

        return mask

    def get_item(self, index):
        batch = super().get_item(index)
        landmark = self.landmarks[index]
        batch["landmarks"] = landmark
        return batch


@DATASET_REGISTRY.register_module
class FDFDensePoseDataset(FDFDataset):

    def __init__(
            self,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.load_landmarks()
        self.load_mask()

    def load_mask(self):
        filepath = self.dirpath.joinpath("mask", f"{self.imsize}.npy")
        assert filepath.is_file(),\
            f"Did not find mask at: {filepath}"
        masks = np.load(filepath)
        assert len(masks) == len(self)
        assert masks.dtype == np.bool
        self.masks = masks

    def get_item(self, index):
        batch = super().get_item(index)
        landmark = self.landmarks[index]
        batch["landmarks"] = landmark
        return batch

    def get_mask(self, idx):
        return self.masks[idx]


@DATASET_REGISTRY.register_module
class FDFRetinaNetPose(FDFDataset):

    def filter_images(self):
        super().filter_images()
        discared_images_fp = self.dirpath.joinpath("discared_images.txt")
        with open(discared_images_fp, "r") as f:
            discared_indices = f.readlines()
            discared_indices = [
                int(_.strip()) for _ in discared_indices
                if _.strip() != ""]
            discared_indices = set(discared_indices)
        keep_indices = set(range(len(self.image_paths)))
        keep_indices = keep_indices.difference(discared_indices)
        self._keep_indices = keep_indices
        self.image_paths = [self.image_paths[idx] for idx in keep_indices]

    def load_landmarks(self):
        filepath = self.dirpath.joinpath("retinanet_landmarks.npy")
        assert filepath.is_file(), \
            f"Did not find landmarks at: {filepath}"
        landmarks = np.load(filepath).reshape(-1, 5, 2)
        landmarks = landmarks.astype(np.float32)
        landmarks = landmarks[np.array(list(self._keep_indices)), ::]
        self.landmarks = landmarks[:len(self)]
        assert len(self.landmarks) == len(self),\
            f"Number of images: {len(self)}, landmarks: {len(landmarks)}"

    def load_bounding_box(self):
        filepath = self.dirpath.joinpath(
            "bounding_box", f"{self.imsize}.torch")
        bbox = load_torch(filepath)
        bbox = bbox[torch.tensor(list(self._keep_indices)), ::]
        self.bounding_boxes = bbox[:len(self)]
        assert len(self.bounding_boxes) == len(self)
