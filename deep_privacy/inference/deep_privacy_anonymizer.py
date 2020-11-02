import numpy as np
import torch
import deep_privacy.torch_utils as torch_utils
import cv2
import pathlib
import typing
from deep_privacy.detection.detection_api import ImageAnnotation
from .anonymizer import Anonymizer
from . import infer


def batched_iterator(batch, batch_size):
    k = list(batch.keys())[0]
    num_samples = len(batch[k])
    num_batches = int(np.ceil(num_samples / batch_size))
    for idx in range(num_batches):
        start = batch_size * idx
        end = start + batch_size
        yield {
            key: torch_utils.to_cuda(arr[start:end])
            for key, arr in batch.items()
        }


class DeepPrivacyAnonymizer(Anonymizer):

    def __init__(self, generator, batch_size, save_debug,
                 fp16_inference: bool,
                 truncation_level=5, **kwargs):
        super().__init__(**kwargs)
        self.inference_imsize = self.cfg.models.max_imsize
        self.batch_size = batch_size
        self.pose_size = self.cfg.models.pose_size
        self.generator = generator
        self.truncation_level = truncation_level
        self.save_debug = save_debug
        self.fp16_inference = fp16_inference
        self.debug_directory = pathlib.Path(".debug", "inference")
        self.debug_directory.mkdir(exist_ok=True, parents=True)

    @torch.no_grad()
    def _get_face(self, batch):
        keys = ["condition", "mask", "landmarks", "z"]
        forward = [batch[k] for k in keys]
#        print([x.shape for x in forward])
        with torch.cuda.amp.autocast(enabled=self.fp16_inference):
            return self.generator(*forward).cpu()

    @torch.no_grad()
    def anonymize_images(self,
                         images: np.ndarray,
                         image_annotations: typing.List[ImageAnnotation]
                         ) -> typing.List[np.ndarray]:
        anonymized_images = []
        for im_idx, image_annotation in enumerate(image_annotations):
            # pre-process
            imsize = self.inference_imsize
            condition = torch.zeros(
                (len(image_annotation), 3, imsize, imsize),
                dtype=torch.float32)
            mask = torch.zeros((len(image_annotation), 1, imsize, imsize))
            landmarks = torch.empty(
                (len(image_annotation), self.pose_size), dtype=torch.float32)
            for face_idx in range(len(image_annotation)):
                face, mask_ = image_annotation.get_face(face_idx, imsize)
                condition[face_idx] = torch_utils.image_to_torch(
                    face, cuda=False, normalize_img=True
                )
                mask[face_idx, 0] = torch.from_numpy(mask_).float()
                kp = image_annotation.aligned_keypoint(face_idx)
                landmarks[face_idx] = kp[:, :self.pose_size]
            img = condition
            condition = condition * mask
            z = infer.truncated_z(
                condition, self.cfg.models.generator.z_shape,
                self.truncation_level)
            batches = dict(
                condition=condition,
                mask=mask,
                landmarks=landmarks,
                z=z,
                img=img
            )
            # Inference
            anonymized_faces = np.zeros((
                len(image_annotation), imsize, imsize, 3), dtype=np.float32)
            for idx, batch in enumerate(
                    batched_iterator(batches, self.batch_size)):
                face = self._get_face(batch)
                face = torch_utils.image_to_numpy(
                    face, to_uint8=False, denormalize=True)
                start = idx * self.batch_size
                anonymized_faces[start:start + self.batch_size] = face
            anonymized_image = image_annotation.stitch_faces(anonymized_faces)
            anonymized_images.append(anonymized_image)
            if self.save_debug:
                num_faces = len(batches["condition"])
                for face_idx in range(num_faces):
                    orig_face = torch_utils.image_to_numpy(
                        batches["img"][face_idx], denormalize=True, to_uint8=True)
                    condition = torch_utils.image_to_numpy(
                        batches["condition"][face_idx],
                        denormalize=True, to_uint8=True)
                    fake_face = anonymized_faces[face_idx]
                    fake_face = (fake_face * 255).astype(np.uint8)
                    to_save = np.concatenate(
                        (orig_face, condition, fake_face), axis=1)
                    filepath = self.debug_directory.joinpath(
                        f"im{im_idx}_face{face_idx}.png")
                    cv2.imwrite(str(filepath), to_save[:, :, ::-1])

        return anonymized_images

    def use_mask(self):
        return self.generator.use_mask
