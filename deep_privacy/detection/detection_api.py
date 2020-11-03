import numpy as np
import typing
import torch
import face_detection
import cv2
from deep_privacy import torch_utils
from deep_privacy.box_utils import clip_box, expand_bbox, cut_face
from . import keypoint_rcnn
from .build import DETECTOR_REGISTRY
from .utils import match_bbox_keypoint


def tight_crop(array):
    mask = array == 0
    x0 = np.argmax(mask.any(axis=0))
    x1 = array.shape[1] - np.argmax((np.flip(mask, axis=1)).any(axis=0))
    y0 = np.argmax(mask.any(axis=1))
    y1 = array.shape[0] - np.argmax((np.flip(mask, axis=0).any(axis=1)))
    return np.array([x0, y0, x1, y1])


def generate_rotation_matrix(
        im, landmark, bbox, detector_cls: str, inverse=False):
    """
        Creates a rotation matrix to align the two eye landmarks to be horizontal.
    """
    if detector_cls == "BaseDetector":
        lm0 = landmark[0]
        lm1 = landmark[1]
    else:
        lm0 = landmark[2]
        lm1 = landmark[1]
    l1 = lm1 - lm0
    l2 = np.array((1, 0)).astype(int)
    alpha = np.arctan2(*l2) - np.arctan2(*l1)
    if inverse:
        alpha = - alpha
    center = bbox[:2] + (bbox[2:] - bbox[:2]) / 2
    center = (center[0], center[1])
    matrix = cv2.getRotationMatrix2D(
        center=center, angle=180 * alpha / np.pi, scale=1)

    # Expand bbox to prevent cutting the cheek
    box = bbox.reshape(-1, 2)
    box = np.pad(box, ((0, 0,), (0, 1)), constant_values=1)
    box = box.dot(matrix.T).reshape(-1)
    x0, y0, x1, y1 = box
    new_y_len = max(x1 - x0, y1 - y0)
    cent = bbox[1] + (bbox[3] - bbox[1]) / 2
    orig = bbox
    bbox = bbox.copy()
    bbox[1] = cent - new_y_len / 2
    bbox[3] = cent + new_y_len / 2
    bbox[1] = max(bbox[1], 0)
    bbox[3] = min(bbox[3], im.shape[1])
    return matrix, orig


class ImageAnnotation:

    def __init__(
            self,
            bbox_XYXY: np.ndarray,
            keypoints: np.ndarray,
            im: np.ndarray,
            detector_cls: str,
            simple_expand: bool,
            align_faces: bool,
            resize_background: bool,
            generator_imsize: int):
        self.align_faces = align_faces
        self.resize_background = resize_background
        self.generator_imsize = generator_imsize
        self.bbox_XYXY = bbox_XYXY
        self.keypoints = keypoints[:, :7, :]
        self.im = im
        self.imshape = im.shape
        self.mask = None
        self._detector_cls = detector_cls
        assert keypoints.shape[2] == 2, f"Shape: {keypoints.shape}"
        assert bbox_XYXY.shape[1] == 4
        self.match()
        self.preprocess()
        self.simple_expand = simple_expand

    def preprocess(self):
        if self.align_faces:
            self.rotation_matrices = np.zeros((len(self), 2, 3))
            for face_idx in range(len(self)):
                rot_matrix, new_bbox = generate_rotation_matrix(
                    self.im, self.keypoints[face_idx],
                    self.bbox_XYXY[face_idx], self._detector_cls)
                self.rotation_matrices[face_idx] = rot_matrix
                self.bbox_XYXY[face_idx] = new_bbox

    def match(self):
        self.bbox_XYXY, self.keypoints = match_bbox_keypoint(
            self.bbox_XYXY, self.keypoints
        )
        assert self.bbox_XYXY.shape[0] == self.keypoints.shape[0]

    def get_expanded_bbox(self, face_idx):
        assert face_idx < len(self)
        tight_bbox = self.bbox_XYXY[face_idx]
        expanded_bbox = expand_bbox(
            tight_bbox,
            self.im.shape,
            simple_expand=self.simple_expand,
            default_to_simple=True,
            expansion_factor=0.35
        )
        width = expanded_bbox[2] - expanded_bbox[0]
        height = expanded_bbox[3] - expanded_bbox[1]
        assert width == height
        return expanded_bbox

    def aligned_keypoint(self, face_idx):
        assert face_idx < len(self)
        keypoint = self.keypoints[face_idx].copy().astype(float)
        if self.align_faces:
            matrix = self.rotation_matrices[face_idx]
            keypoint = np.pad(
                keypoint, ((0, 0), (0, 1)), constant_values=1
            )
            keypoint = keypoint.dot(matrix.T)
        expanded_bbox = self.get_expanded_bbox(face_idx)
        keypoint[:, 0] -= expanded_bbox[0]
        keypoint[:, 1] -= expanded_bbox[1]
        w = expanded_bbox[2] - expanded_bbox[0]
        keypoint /= w
        keypoint[keypoint < 0] = 0
        keypoint[keypoint > 1] = 1
        keypoint = torch.from_numpy(keypoint).view(1, -1)
        return keypoint

    def __repr__(self):
        return f"Image Annotation. BBOX_XYXY: {self.bbox_XYXY.shape}" +\
            f" Keypoints: {self.keypoints.shape}"

    def __len__(self):
        return self.keypoints.shape[0]

    def get_mask(self, idx):
        mask = np.ones(self.im.shape[:2], dtype=np.bool)
        x0, y0, x1, y1 = self.bbox_XYXY[idx]
        mask[y0:y1, x0:x1] = 0
        return mask

    def get_cut_mask(self, idx, imsize):
        box_exp = self.get_expanded_bbox(idx)
        boxes = self.bbox_XYXY[idx].copy().astype(np.float32)
        boxes[[0, 2]] -= box_exp[0]
        boxes[[1, 3]] -= box_exp[1]
        resize_factor = imsize / (box_exp[2] - box_exp[0])
        boxes *= resize_factor
        boxes = boxes.astype(int)
        mask = np.ones((imsize, imsize), dtype=np.bool)
        x0, y0, x1, y1 = boxes
        mask[y0:y1, x0:x1] = 0
        return mask

    def get_face(self, face_idx: int, imsize):
        assert face_idx < len(self)
        bbox = self.get_expanded_bbox(face_idx)
        im = self.im
        if self.align_faces:
            rot_matrix = self.rotation_matrices[face_idx]
            im = cv2.warpAffine(
                im, M=rot_matrix, dsize=(self.im.shape[1], self.im.shape[0]))
        face = cut_face(im, bbox, simple_expand=self.simple_expand)
        if imsize is not None:
            face = cv2.resize(face, (imsize, imsize),
                              interpolation=cv2.INTER_CUBIC)
        mask = self.get_cut_mask(face_idx, imsize)
        return face, mask

    def paste_face(self, face_idx, face):
        """
            Rotates the original image, pastes in the rotated face, then inverse rotate.
        """
        im = self.im / 255
        if self.align_faces:
            matrix = self.rotation_matrices[face_idx]
            im = cv2.warpAffine(
                im, M=matrix, dsize=(self.im.shape[1], self.im.shape[0]))

        bbox = self.bbox_XYXY[face_idx].copy()
        exp_bbox = self.get_expanded_bbox(face_idx)
        bbox[[0, 2]] -= exp_bbox[0]
        bbox[[1, 3]] -= exp_bbox[1]

        x0, y0, x1, y1 = self.get_expanded_bbox(face_idx)

        # expanded bbox might go outside of image.
        im[max(0, y0):min(y1, im.shape[0]),
           max(0, x0):min(x1, im.shape[1])] = face[
            max(-y0, 0): min(face.shape[0], face.shape[1] - (y1 - im.shape[0])),
            max(-x0, 0): min(face.shape[0], face.shape[0] - (x1 - im.shape[1]))]

        if self.align_faces:
            matrix, _ = generate_rotation_matrix(
                self.im, self.keypoints[face_idx], self.bbox_XYXY[face_idx],
                self._detector_cls,
                inverse=True
            )
            im = cv2.warpAffine(
                im, M=matrix, dsize=(self.im.shape[1], self.im.shape[0]))
        return im

    def stitch_faces(self, anonymized_faces):
        """
            Copies the generated face(s) to the original face
            Make sure that an already anonymized face is not overwritten.
        """
        im = self.im.copy()
        mask_not_filled = np.ones_like(im, dtype=bool)
        for face_idx, face in enumerate(anonymized_faces):
            orig_bbox = self.bbox_XYXY[face_idx]
            expanded_bbox = self.get_expanded_bbox(face_idx)
            orig_face_shape = (
                expanded_bbox[2] - expanded_bbox[0],
                expanded_bbox[3] - expanded_bbox[1]
            )
            face = cv2.resize(face, orig_face_shape)
            inpainted_im = self.paste_face(face_idx, face) * 255
            mask_ = cut_face(mask_not_filled, orig_bbox)
            x0, y0, x1, y1 = orig_bbox
            im[y0:y1, x0:x1][mask_] = inpainted_im[y0:y1, x0:x1][mask_]
            mask_not_filled[y0:y1, x0:x1] = 0
            if self.resize_background:
                face = cut_face(im, expanded_bbox, pad_im=False)
                orig_shape = face.shape[:2][::-1]
                face = cv2.resize(face, (self.generator_imsize*2, self.generator_imsize*2))
                
                x0, y0, x1, y1 = clip_box(expanded_bbox, im)
                im[y0:y1, x0:x1] = cv2.resize(face, orig_shape)
        return im


@DETECTOR_REGISTRY.register_module
class BaseDetector:

    def __init__(self, face_detector_cfg: dict, simple_expand: bool,
                 align_faces: bool, resize_background: bool,
                 generator_imsize: int, *args, **kwargs):
        self.face_detector = face_detection.build_detector(
            **face_detector_cfg,
            device=torch_utils.get_device()
        )
        self.simple_expand = simple_expand
        self.align_faces = align_faces
        self.resize_background = resize_background
        self.generator_imsize = generator_imsize
        if self.__class__.__name__ == "BaseDetector":
            assert face_detector_cfg.name == "RetinaNetResNet50"

    def post_process_detections(
            self,
            images: typing.List[np.ndarray],
            im_bboxes: typing.List[np.ndarray],
            keypoints: typing.List[np.ndarray]):
        image_annotations = []
        for im_idx, im in enumerate(images):
            annotation = ImageAnnotation(
                im_bboxes[im_idx], keypoints[im_idx], im,
                self.__class__.__name__,
                self.simple_expand,
                self.align_faces,
                self.resize_background,
                self.generator_imsize
            )
            image_annotations.append(annotation)
        return image_annotations

    def get_detections(self,
                       images: typing.List[np.ndarray],
                       im_bboxes: typing.List[np.ndarray] = None
                       ) -> typing.List[ImageAnnotation]:
        im_bboxes = []
        keypoints = []
        if im_bboxes is None or len(im_bboxes) == 0:
            for im in images:
                boxes, keyps = self.face_detector.batched_detect_with_landmarks(
                    im[None])
                boxes = boxes[0][:, :4]
                im_bboxes.append(boxes.astype(int))
                keypoints.append(keyps[0])
        return self.post_process_detections(images, im_bboxes, keypoints)


@DETECTOR_REGISTRY.register_module
class RCNNDetector(BaseDetector):

    def __init__(self, keypoint_threshold: float, rcnn_batch_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keypoint_detector = keypoint_rcnn.RCNNKeypointDetector(
            keypoint_threshold, rcnn_batch_size)
        self.keypoint_threshold = keypoint_threshold

    def detect_faces(self, images, im_bboxes):
        if im_bboxes is None or len(im_bboxes) == 0:
            im_bboxes = []
            for im in images:
                boxes = self.face_detector.batched_detect(im[None])
                boxes = boxes[0][:, :4]
                im_bboxes.append(boxes.astype(int))
        return im_bboxes

    def get_detections(self, images, im_bboxes=None):
        im_bboxes = self.detect_faces(images, im_bboxes)
        keypoints = self.keypoint_detector.batch_detect_keypoints(images)
        return self.post_process_detections(images, im_bboxes, keypoints)
