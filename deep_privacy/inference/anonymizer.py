import moviepy.editor as mp
import tqdm
import os
import cv2
import numpy as np
from deep_privacy.visualization import utils as vis_utils
from deep_privacy.inference import infer
from deep_privacy.detection import detection_api
from deep_privacy.inference import utils as inference_utils


class Anonymizer:

    def __init__(self, face_threshold=.1, keypoint_threshold=.3, *args, **kwargs):
        super().__init__()
        self.face_threshold = face_threshold
        self.keypoint_threshold = keypoint_threshold

    def anonymize_images(self, images, im_keypoints, im_bboxes):
        raise NotImplementedError

    def anonymize_folder(self, folder_path, save_path, im_bboxes=None):
        if folder_path.endswith("/"):
            folder_path = folder_path[:-1]
        image_paths = infer.get_images_recursive(folder_path)

        relative_paths = [impath[len(folder_path)+1:]
                          for impath in image_paths]
        save_paths = [os.path.join(save_path, impath)
                      for impath in relative_paths]
        self.anonymize_image_paths(image_paths,
                                   save_paths,
                                   im_bboxes=im_bboxes)

    def anonymize_image_paths(self, image_paths, save_paths, im_bboxes=None):
        images = [cv2.imread(p)[:, :, ::-1] for p in image_paths]
        im_bboxes, im_keypoints = detection_api.batch_detect_faces_with_keypoints(
            images, im_bboxes=im_bboxes,
            keypoint_threshold=self.keypoint_threshold,
            face_threshold=self.face_threshold
        )

        anonymized_images = self.anonymize_images(images,
                                                  im_keypoints=im_keypoints,
                                                  im_bboxes=im_bboxes)

        for image_idx, (new_path, anon_im) in enumerate(zip(save_paths,
                                                            anonymized_images)):
            os.makedirs(os.path.dirname(new_path), exist_ok=True)

            annotated_im = vis_utils.draw_faces_with_keypoints(
                images[image_idx],
                im_bboxes[image_idx],
                im_keypoints[image_idx]
            )
            cv2.imwrite(new_path, anon_im[:, :, ::-1])

            to_save = np.concatenate((annotated_im, anon_im), axis=1)
            debug_impath = new_path.split(".")[0] + "_detected_left_anonymized_right.jpg"
            cv2.imwrite(debug_impath, to_save[:, :, ::-1])

    def anonymize_video(self, video_path, target_path,
                        start_frame=None,
                        end_frame=None,
                        with_keypoints=False,
                        anonymize_source=False,
                        max_face_size=1.0):
        # Read original video
        original_video = mp.VideoFileClip(video_path)
        fps = original_video.fps
        total_frames = int(original_video.duration * original_video.fps)
        start_frame = 0 if start_frame is None else start_frame
        end_frame = total_frames if end_frame is None else end_frame
        assert start_frame <= end_frame, f"Start frame{start_frame} has to be smaller than end frame {end_frame}"
        assert end_frame <= total_frames, f"End frame ({end_frame}) is larger than number of frames {total_frames}"
        subclip = original_video.subclip(start_frame/fps, end_frame/fps)
        print("="*80)
        print("Anonymizing video.")
        print(
            f"Duration: {original_video.duration}. Total frames: {total_frames}, FPS: {fps}")
        print(
            f"Anonymizing from: {start_frame}({start_frame/fps}), to: {end_frame}({end_frame/fps})")

        frames = list(tqdm.tqdm(subclip.iter_frames(), desc="Reading frames",
                                total=end_frame - start_frame))
        if with_keypoints:
            im_bboxes, im_keypoints = detection_api.batch_detect_faces_with_keypoints(
                frames)
            im_bboxes, im_keypoints = inference_utils.filter_image_bboxes(
                im_bboxes, im_keypoints,
                [im.shape for im in frames],
                max_face_size,
                filter_type="width"
            )
            anonymized_frames = self.anonymize_images(frames,
                                                      im_keypoints,
                                                      im_bboxes)
        else:
            im_bboxes = detection_api.batch_detect_faces(frames,
                                                         self.face_threshold)
            im_keypoints = None
            anonymized_frames = self.anonymize_images(frames, im_bboxes)

        def make_frame(t):
            frame_idx = int(round(t * original_video.fps))
            anonymized_frame = anonymized_frames[frame_idx]
            orig_frame = frames[frame_idx]
            orig_frame = vis_utils.draw_faces_with_keypoints(
                orig_frame, im_bboxes[frame_idx], im_keypoints[frame_idx],
                radius=None,
                black_out_face=anonymize_source)
            return np.concatenate((orig_frame, anonymized_frame), axis=1)

        anonymized_video = mp.VideoClip(make_frame)
        anonymized_video.duration = (end_frame - start_frame) / fps
        anonymized_video.fps = fps
        to_concatenate = []
        if start_frame != 0:
            to_concatenate.append(original_video.subclip(0, start_frame/fps))
        to_concatenate.append(anonymized_video)
        if end_frame != total_frames:
            to_concatenate.append(original_video.subclip(end_frame/fps, total_frames/fps))
        anonymized_video = mp.concatenate(to_concatenate)

        anonymized_video.audio = original_video.audio
        print("Anonymized video stats.")
        total_frames = int(anonymized_video.duration * anonymized_video.fps)
        print(f"Duration: {anonymized_video.duration}. Total frames: {total_frames}, FPS: {fps}")
        print(f"Anonymizing from: {start_frame}({start_frame/fps}), to: {end_frame}({end_frame/fps})")

        anonymized_video.write_videofile(target_path, fps=original_video.fps,
                                         audio_codec='aac')
