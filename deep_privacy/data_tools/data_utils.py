import torch
from deep_privacy.models.utils import get_transition_value


class DataPrefetcher():

    def __init__(self, loader, pose_size, dataset):
        self.pool = torch.nn.AvgPool2d(2, 2)
        self.original_loader = loader
        self.stream = torch.cuda.Stream()
        self.pose_size = pose_size
        self.i = 0
        self.dataset = dataset

    def preload(self):
        try:
            self.next_image, self.next_condition, self.next_landmark = next(
                self.loader)
        except StopIteration:
            self.next_image = None
            self.next_condition = None
            self.next_landmark = None
            return
        with torch.cuda.stream(self.stream):
            self.next_image = self.next_image.cuda(non_blocking=True).float()
            self.next_condition = self.next_condition.cuda(
                non_blocking=True).float()
            self.next_landmark = self.next_landmark.cuda(non_blocking=True)

            self.next_image = interpolate_image(self.pool,
                                                self.next_image,
                                                self.transition_variable)
            self.next_condition = interpolate_image(self.pool,
                                                    self.next_condition,
                                                    self.transition_variable)
            self.next_image = self.next_image / 255
            self.next_image = self.next_image*2 - 1

            self.next_condition = self.next_condition / 255
            self.next_condition = self.next_condition*2 - 1

    def __len__(self):
        return len(self.original_loader)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        next_image = self.next_image
        if next_image is None:
            raise StopIteration
        next_condition = self.next_condition
        next_landmark = self.next_landmark
        self.preload()
        return next_image, next_condition, next_landmark[:, :self.pose_size]

    def __iter__(self):
        self.loader = iter(self.original_loader)
        self.preload()
        return self

    def update_next_transition_variable(self, transition_variable):
        self.transition_variable = transition_variable
        self.dataset.transition_variable = self.transition_variable


def interpolate_image(pool, images, transition_variable):
    assert images.max() > 1
    y = pool(images) // 1
    y = torch.nn.functional.interpolate(y, scale_factor=2)
    images = get_transition_value(y, images, transition_variable)
    return images


def denormalize_img(image):
    image = (image+1)/2
    image = torch.clamp(image, 0, 1)
    return image
