import moviepy.editor as mp
import cv2
import numpy as np
import pathlib
import typing
from deep_privacy.visualization import utils as vis_utils
from deep_privacy.detection import build_detector, ImageAnnotation


class Anonymizer:

    def __init__(self,
                 cfg,
                 detector_cfg,
                 *args, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.detector = build_detector(
            detector_cfg, generator_imsize=cfg.models.max_imsize)

    def anonymize(self, im):
        return self.detect_and_anonymize_images([im])[0]

    def get_detections(
            self,
            images: typing.List[np.ndarray],
            im_bboxes: typing.List[np.ndarray] = None,
    ) -> typing.List[ImageAnnotation]:
        image_annotations = self.detector.get_detections(
            images, im_bboxes=im_bboxes,
        )

        return image_annotations

    def detect_and_anonymize_images(
            self,
            images: typing.List[np.ndarray],
            im_bboxes: typing.List[np.ndarray] = None,
            return_annotations: bool = False):
        image_annotations = self.get_detections(
            images, im_bboxes)

        anonymized_images = self.anonymize_images(
            images,
            image_annotations
        )
        if return_annotations:
            return anonymized_images, image_annotations
        return anonymized_images

    def anonymize_images(self,
                         images: np.ndarray,
                         image_annotations: typing.List[ImageAnnotation]):
        raise NotImplementedError

    def anonymize_image_paths(self,
                              image_paths: typing.List[pathlib.Path],
                              save_paths: typing.List[pathlib.Path],
                              im_bboxes=None):
        images = [cv2.imread(str(p))[:, :, ::-1] for p in image_paths]
        anonymized_images, image_annotations = self.detect_and_anonymize_images(
            images, im_bboxes, return_annotations=True)

        for image_idx, (new_path, anon_im) in enumerate(
                zip(save_paths, anonymized_images)):
            new_path.parent.mkdir(exist_ok=True, parents=True)
            annotation = image_annotations[image_idx]
            annotated_im = images[image_idx]
            for face_idx in range(len(annotation)):
                annotated_im = annotated_im * annotation.get_mask(face_idx)[:, :, None]
            annotated_im = vis_utils.draw_faces_with_keypoints(
                annotated_im,
                annotation.bbox_XYXY,
                annotation.keypoints
            )
            cv2.imwrite(str(new_path), anon_im[:, :, ::-1])
            print("Saving to:", new_path)

            to_save = np.concatenate((annotated_im, anon_im), axis=1)
            new_name = new_path.stem + "_detected_left_anonymized_right.jpg"
            debug_impath = new_path.parent.joinpath(new_name)
            cv2.imwrite(str(debug_impath), to_save[:, :, ::-1])

    def anonymize_video(self, video_path: pathlib.Path,
                        target_path: pathlib.Path,
                        start_time=None,
                        end_time=None):
        # Read original video
        original_video = mp.VideoFileClip(str(video_path))
        fps = original_video.fps
        total_frames = int(original_video.duration * original_video.fps)
        start_time = 0 if start_time is None else start_time
        end_time = original_video.duration if end_time is None else end_time
        assert start_time <= end_time, f"Start frame{start_time} has to be smaller than end frame {end_time}"
        assert end_time <= original_video.duration,\
            f"End frame ({end_time}) is larger than number of frames {original_video.duration}"
        print(*[
            "=" * 80,
            "Anonymizing video.",
            f"Duration: {original_video.duration}. Total frames: {total_frames}, FPS: {fps}",
            f"Anonymizing from: {start_time}({start_time*fps})s, to: {end_time}({end_time*fps})s"
        ], sep="\n")
        self.frame_idx = 0

        start = original_video.subclip(0, start_time)
        end = original_video.subclip(end_time)
        anonymized_video = original_video.subclip(start_time, end_time)
        anonymized_video = anonymized_video.fl_image(self.anonymize)
        anonymized_video = mp.concatenate_videoclips([start, anonymized_video, end])


        anonymized_video.write_videofile(
            str(target_path),
            fps=original_video.fps,
            audio_codec='aac')
