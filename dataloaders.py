import torch
import os
import utils
import glob


def load_torch_files(dirpath):
    images = []
    files = glob.glob(os.path.join(dirpath, "*.torch"))
    files.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
    assert len(files) > 0, "Empty directory: " + dirpath
    for fpath in files:
        assert os.path.isfile(fpath), "Is not file: " + fpath
        ims = torch.load(fpath)
        assert ims.dtype == torch.float32, "Was: {}".format(ims.dtype)
        images.append(ims)
    images = torch.cat(images, dim=0)
    return images


def load_ffhq_condition(batch_size, imsize=128):
    return ConditionedCelebADataset("data/ffhq_torch", imsize, batch_size, landmarks_total=False)


def load_celeba_condition(batch_size, imsize=128):
    return ConditionedCelebADataset("data/celeba_torch", imsize, batch_size)


def bounding_box_data_augmentation(bounding_boxes, imsize, percentage):
    # Data augment width and height by percentage of width.
    # Bounding box will preserve its center.
    shrink_percentage = torch.zeros(
        (bounding_boxes.shape[0])).float().uniform_(-percentage, percentage)
    width = (bounding_boxes[:, 2] - bounding_boxes[:, 0]).float()
    height = (bounding_boxes[:, 3] - bounding_boxes[:, 1]).float()

    # Can change 10% in each direction
    width_diff = shrink_percentage * width
    height_diff = shrink_percentage * height
    bounding_boxes[:, 0] -= width_diff.long()
    bounding_boxes[:, 1] -= height_diff.long()
    bounding_boxes[:, 2] += width_diff.long()
    bounding_boxes[:, 3] += height_diff.long()
    # Ensure that bounding box is within image
    bounding_boxes[:, 0][bounding_boxes[:, 0] < 0] = 0
    bounding_boxes[:, 1][bounding_boxes[:, 1] < 0] = 0
    bounding_boxes[:, 2][bounding_boxes[:, 2] > imsize] = imsize
    bounding_boxes[:, 3][bounding_boxes[:, 3] > imsize] = imsize
    return bounding_boxes


def bbox_to_coordinates(bounding_boxes):
    x0 = bounding_boxes[:, 0]
    y0 = bounding_boxes[:, 0]
    x1 = x0 + bounding_boxes[:, 2]
    y1 = y0 + bounding_boxes[:, 3]
    bounding_boxes[:, 2] = x1
    bounding_boxes[:, 3] = y1
    return bounding_boxes


def cut_bounding_box(condition, bounding_boxes):
    x0 = bounding_boxes[:, 0]
    y0 = bounding_boxes[:, 1]
    x1 = bounding_boxes[:, 2]
    y1 = bounding_boxes[:, 3]
    for i in range(condition.shape[0]):
        previous_image = condition[i, :, y0[i]:y1[i], x0[i]:x1[i]]
        mean = previous_image.mean()
        std = previous_image.std()
        if previous_image.shape[1] == 0 or previous_image.shape[2] == 0:
            continue
        previous_image[:, :, :] = utils.truncated_normal(mean,
                                                         std,
                                                         previous_image.shape,
                                                         previous_image.max(),
                                                         previous_image.min())
    return condition


class ConditionedCelebADataset:

    def __init__(self, dirpath, imsize, batch_size, landmarks_total=False):
        self.images = load_torch_files(
            os.path.join(dirpath, "original", str(imsize)))
        bounding_box_filepath = os.path.join(
            dirpath, "bounding_box", "{}.torch".format(imsize))
        assert os.path.isfile(bounding_box_filepath), "Did not find the bounding box data. Looked in: {}".format(
            bounding_box_filepath)
        self.bounding_boxes = torch.load(bounding_box_filepath).long()

        if landmarks_total:
            landmark_filepath = os.path.join(
                dirpath, "landmarks_total", "{}.torch".format(imsize))
        else:
            landmark_filepath = os.path.join(
                dirpath, "landmarks", "{}.torch".format(imsize))
        assert os.path.isfile(
            landmark_filepath), "Did not find the landmark data. Looked in: {}".format(landmark_filepath)
        self.landmarks = torch.load(landmark_filepath).float() / imsize

        assert self.landmarks.shape[0] == self.bounding_boxes.shape[0]
        assert self.bounding_boxes.shape[0] == self.images.shape[0], "The number \
            of samples of images doesn't match number of bounding boxes. Images: {}, bbox: {}".format(self.images.shape[0], self.bounding_boxes.shape[0])
        expected_imshape = (3, imsize, imsize)
        assert self.images.shape[1:] == expected_imshape, "Shape was: {}. Expected: {}".format(
            self.images.shape[1:], expected_imshape)
        print("Dataset loaded. Number of samples:", self.images.shape)
        self.n_samples = self.images.shape[0]
        self.batch_size = batch_size
        self.indices = torch.LongTensor(self.batch_size)
        self.batches_per_epoch = self.n_samples // self.batch_size
        self.validation_size = int(self.n_samples*0.05)
        self.imsize = imsize

    def __len__(self):
        return self.images.shape[0] // self.batch_size

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n > self.batches_per_epoch:
            raise StopIteration
        self.n += 1
        indices = self.indices.random_(
            0, self.n_samples - self.validation_size)
        images = self.images[indices]
        bounding_boxes = self.bounding_boxes[indices]
        landmarks = self.landmarks[indices]  # only left eye, right eye, nose
        bounding_boxes = bounding_box_data_augmentation(
            bounding_boxes, self.imsize, 0.2)
        condition = images.clone()
        condition = cut_bounding_box(condition, bounding_boxes)
        to_flip = torch.rand(self.batch_size) > .5
        try:
            images[to_flip] = utils.flip_horizontal(images[to_flip])
            condition[to_flip] = utils.flip_horizontal(condition[to_flip])
            landmarks[to_flip][:, range(0, landmarks.shape[1], 2)] = 1 - \
                landmarks[to_flip][:, range(
                    0, landmarks.shape[1], 2)]  # Flip the x-values
        except Exception as e:
            print("Could not flip images.", e)
        return images, condition, landmarks

    def validation_set_generator(self):
        validation_iters = self.validation_size // self.batch_size
        validation_offset = self.n_samples - self.validation_size
        for i in range(validation_iters):
            start_idx = validation_offset + i*self.batch_size
            end_idx = validation_offset + (i+1)*self.batch_size
            images = self.images[start_idx:end_idx]
            bounding_boxes = self.bounding_boxes[start_idx:end_idx]
            landmarks = self.landmarks[start_idx:end_idx]
            condition = images.clone()
            condition = cut_bounding_box(condition, bounding_boxes)
            yield images, condition, landmarks
