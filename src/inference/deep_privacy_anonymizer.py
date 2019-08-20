import numpy as np
import torch
import src.torch_utils as torch_utils
import tqdm
import os
import cv2
from .anonymizer import Anonymizer
from collections import defaultdict
from . import infer


class DeepPrivacyAnonymizer(Anonymizer):

    def __init__(self, generator, batch_size, use_static_z, save_debug=True):
        super().__init__()
        self.inference_imsize = generator.current_imsize
        self.batch_size = batch_size
        self.pose_size = generator.num_poses * 2
        self.generator = generator
        self.use_static_z = use_static_z
        if self.use_static_z:
            self.static_z = self.generator.generate_latent_variable(self.batch_size, "cuda", torch.float32).zero_()
        self.save_debug = save_debug
        self.debug_directory = os.path.join(".debug", "inference")
        if self.save_debug:
            os.makedirs(self.debug_directory, exist_ok=True)

    def anonymize_images(self, images, im_keypoints, im_bboxes):
        face_info = self.pre_process_faces(images, im_keypoints, im_bboxes)
        generated_faces = self.anonymize_faces(face_info)
        anonymized_images = self.post_process(face_info, generated_faces, images)
        if self.save_debug:
            self.save_debug_images(face_info, generated_faces)
        return anonymized_images

    def save_debug_images(self, face_info, generated_faces):
        for face_idx, info in face_info.items():
            torch_input = info["torch_input"].squeeze(0)
            generated_face = generated_faces[face_idx]
            torch_input = torch_utils.image_to_numpy(torch_input,
                                                     to_uint8=True,
                                                     denormalize=True)
            generated_face = torch_utils.image_to_numpy(generated_face,
                                                        to_uint8=True,
                                                        denormalize=True)
            print(torch_input.shape, generated_face.shape)
            to_save = np.concatenate((torch_input, generated_face), axis=1)
            filepath = os.path.join(self.debug_directory, f"face_{face_idx}.jpg")
            cv2.imwrite(filepath, to_save[:, :, ::-1])

    def pre_process_faces(self, images, im_keypoints, im_bboxes):
        face_info = {}
        face_idx = 0
        for im_idx in range(len(images)):
            im = images[im_idx].copy()
            face_keypoints = im_keypoints[im_idx]

            face_bboxes = im_bboxes[im_idx]
            for keypoint, bbox in zip(face_keypoints, face_bboxes):
                res = infer.pre_process(im.copy(), keypoint, bbox,
                                        self.inference_imsize, cuda=False)
                if res is None:
                    continue
                torch_input, keypoint, expanded_bbox, new_bbox = res
                face_info[face_idx] = {
                    "im_idx": im_idx,
                    "face_bbox": bbox,
                    "torch_input": torch_input,
                    "translated_keypoint": keypoint,
                    "expanded_bbox": expanded_bbox
                }
                face_idx += 1
        if len(face_info) == 0:
            return face_info
        return face_info
    
    def anonymize_faces(self, face_info):
        torch_faces = torch.empty((len(face_info), 3, self.inference_imsize, self.inference_imsize),
                                  dtype=torch.float32)
        torch_keypoints = torch.empty((len(face_info), self.pose_size),
                                      dtype=torch.float32)
        for face_idx, face in face_info.items():
            torch_faces[face_idx] = face["torch_input"]
            torch_keypoints[face_idx] = face["translated_keypoint"]
        
        num_batches = int(np.ceil(len(face_info) / self.batch_size))
        results = []
        with torch.no_grad():
            for batch_idx in tqdm.trange(num_batches, desc="Anonyimizing faces"):
                im = torch_faces[batch_idx*self.batch_size:
                                 (batch_idx+1)*self.batch_size]
                keypoints = torch_keypoints[batch_idx*self.batch_size:
                                            (batch_idx+1)*self.batch_size]
                im = torch_utils.to_cuda(im)
                keypoints = torch_utils.to_cuda(keypoints)
                if self.use_static_z:
                    z = self.static_z.clone()
                    z = z[:len(im)]
                    generated_faces = self.generator(im, keypoints, z)
                else:
                    generated_faces = self.generator(im, keypoints)
                results.append(generated_faces.cpu())

        generated_faces = torch.cat(results)
        return generated_faces
    
    def post_process(self, face_info, generated_faces, images):
        anonymized_images = [im.copy() for im in images]
        im_to_face_idx = defaultdict(list)
        for face_idx, info in face_info.items():
            im_idx = info["im_idx"]
            im_to_face_idx[im_idx].append(face_idx)
        
        for im_idx, face_indices in tqdm.tqdm(im_to_face_idx.items(), desc="Post-processing"):
            im = anonymized_images[im_idx]
            replaced_mask = np.ones_like(im).astype("bool")
            for face_idx in face_indices:
                generated_face = generated_faces[face_idx:face_idx+1]
                expanded_bbox = face_info[face_idx]["expanded_bbox"]
                original_bbox = face_info[face_idx]["face_bbox"]
                im = infer.post_process(im, generated_face, expanded_bbox,
                                        original_bbox, replaced_mask)
            anonymized_images[im_idx] = im
        return anonymized_images


if __name__ == "__main__":
    generator, imsize, source_path, image_paths, save_path = infer.read_args()
    a = DeepPrivacyAnonymizer(generator, 128, use_static_z=True)
    a.anonymize_video("test_examples/video/selfie.mp4", "test_examples/video/selfie.mp4", 0, None, True)