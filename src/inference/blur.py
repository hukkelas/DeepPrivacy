import tqdm
import cv2
import numpy as np
from src.inference.anonymizer import Anonymizer


def is_height_larger(bbox, image_shape, max_face_size):
    x0, y0, x1, y1 = bbox
    face_size = (y1 - y0) / image_shape[0]
    return face_size >= max_face_size


class SimpleAnonymizer(Anonymizer):

    def __init__(self, *args, **kwargs):
        super().__init__(kwargs)

    def anonymize_face(self, face):
        raise NotImplementedError

    def anonymize_images(self, images, im_bboxes, im_keypoints=None, max_face_size=1.0):
        anonymized_images = []
        for im_idx, im in enumerate(tqdm.tqdm(images, desc="Anonymizing images")):
            anonymized_image = im.copy()
            bboxes = im_bboxes[im_idx]

            for bbox in bboxes:
                x0, y0, x1, y1 = [int(_) for _ in bbox]
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(im.shape[1], x1), min(im.shape[0], y1)
                if is_height_larger(bbox, anonymized_image.shape, max_face_size):
                    continue
                if y1 - y0 <= 0 or x1 - x0 <= 0: continue
                face = im[y0:y1, x0:x1]
                if face.shape[0] == 0 or face.shape[1] == 0: continue
                orig_shape = face.shape
                face = self.anonymize_face(face)
                assert orig_shape == face.shape, f"Did not return equal sized face."

                anonymized_image[y0:y1, x0:x1] = face

            anonymized_images.append(anonymized_image)
        return anonymized_images


class PixelationAnonymizer(SimpleAnonymizer):

    def __init__(self, pixelation_size, **kwargs):
        super().__init__(kwargs)
        self.pixelation_size = pixelation_size
        print("Pixelation initialize!")

    def anonymize_face(self, face):
        orig_size = face.shape
        face = cv2.resize(face, (self.pixelation_size,
                                 self.pixelation_size))
        face = cv2.resize(face, (orig_size[1], orig_size[0]))
        return face


class BlurAnonymizer(SimpleAnonymizer):

    def __init__(self, blur_type, **kwargs):
        super().__init__(kwargs)
        self.blur_type = blur_type
        print("Blur anonymizer initialize!")

    def anonymize_face(self, face):
        if self.blur_type == "heavy_blur":
            ksize = int(0.3 * face.shape[1])
            ksize = max(1, ksize)
            face = cv2.blur(face, (ksize, ksize))
            return face
        elif self.blur_type == "gaussian_blur":
            face = cv2.GaussianBlur(face, (9, 9), sigmaX=3, sigmaY=3)
            return face
        else:
            raise AttributeError(f'Undefined blur type: {self.blur_type}')


class BlackOutAnonymizer(SimpleAnonymizer):

    def __init__(self, *args, **kwargs):
        super().__init__(kwargs)
        print("BlackOut anonymizer initialize!")

    def anonymize_face(self, face):
        return np.random.normal(
            loc=face.mean(), scale=face.std(),
            size=face.shape
        )
