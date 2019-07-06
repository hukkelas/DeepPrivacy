import torch
import utils
from models.utils import get_transition_value

class DataPrefetcher():

    def __init__(self, loader, transition_variable, pose_size):
        self.pool = torch.nn.AvgPool2d(2, 2)
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload(transition_variable)
        self.pose_size = pose_size

    def preload(self, transition_variable):
        try:
            self.next_image, self.next_condition, self.next_landmark = next(self.loader)
        except StopIteration:
            self.next_image = None
            self.next_condition = None
            self.next_landmark = None
            return
        with torch.cuda.stream(self.stream):
            self.next_image = self.next_image.cuda(non_blocking=True).float()
            self.next_condition = self.next_condition.cuda(non_blocking=True).float()
            self.next_landmark = self.next_landmark.cuda(non_blocking=True)
            
            self.next_image = self.next_image / 255
            self.next_image = self.next_image*2 - 1

            self.next_condition = self.next_condition / 255
            self.next_condition = self.next_condition*2 - 1

            self.next_image = interpolate_image(self.pool,
                                                self.next_image,
                                                transition_variable)
            self.next_condition = interpolate_image(self.pool,
                                                    self.next_condition,
                                                    transition_variable)

    def next(self, transition_variable):
        torch.cuda.current_stream().wait_stream(self.stream)
        next_image = self.next_image
        next_condition = self.next_condition
        next_landmark = self.next_landmark
        self.preload(transition_variable)
        return next_image, next_condition, next_landmark[:, :self.pose_size]


def interpolate_image(pool, images, transition_variable):
    y = pool(images)
    y = torch.nn.functional.interpolate(y, scale_factor=2)

    images = get_transition_value(y, images, transition_variable)
    return images

def preprocess_images(image, transition_variable, pool=torch.nn.AvgPool2d(2, 2)):
    image = image * 2 - 1
    image = interpolate_image(pool, image, transition_variable)
    return image

def denormalize_img(image):
    image = (image+1)/2
    image = utils.clip(image, 0, 1)
    return image