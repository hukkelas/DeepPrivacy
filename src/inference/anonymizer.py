import moviepy.editor as mp
import tqdm
from src.detection import detection_api


class Anonymizer:


    def __init__(self, face_threshold=.3, save_debug=False):
        super().__init__()
        self.face_threshold = face_threshold
        self.save_debug = save_debug

    def anonymize_images(self, images, im_keypoints, im_bboxes):
        raise NotImplementedError

    def anonymize_video(self, video_path, target_path,
                        start_frame=None,
                        end_frame=None):
        original_video = mp.VideoFileClip(video_path)
        fps = original_video.fps
        total_frames = int(original_video.duration * original_video.fps)
        start_frame = 0 if start_frame is None else start_frame
        end_frame = total_frames if end_frame is None else end_frame
        subclip = original_video.subclip(start_frame/fps, end_frame/fps)
        print("Anonymizing video.")
        print(f"Duration: {original_video.duration}. Total frames: {total_frames}, FPS: {fps}")
        print(f"Anonymizing from: {start_frame}({start_frame/fps}), to: {end_frame}({end_frame/fps})")

        frames = list(tqdm.tqdm(subclip.iter_frames(), desc="Reading frames",
                                total=end_frame - start_frame))
        face_bboxes = detection_api.batch_detect_faces(frames,
                                                       self.face_threshold)
        frames = self.anonymize_images(frames, face_bboxes)

        def make_frame(t):
            frame_idx = int(t * original_video.fps)
            return frames[frame_idx]
        anonymized_video = mp.VideoClip(make_frame)
        anonymized_video.duration = (end_frame - start_frame) / fps
        anonymized_video.fps = fps
        anonymized_video = mp.concatenate([
            original_video.subclip(0, start_frame/fps),
            anonymized_video,
            original_video.subclip(end_frame/fps, total_frames/fps)
        ])

        anonymized_video.audio = original_video.audio
        print("Anonymized video stats.")
        total_frames = int(anonymized_video.duration * anonymized_video.fps)
        print(f"Duration: {anonymized_video.duration}. Total frames: {total_frames}, FPS: {fps}")
        print(f"Anonymizing from: {start_frame}({start_frame/fps}), to: {end_frame}({end_frame/fps})")

        anonymized_video.write_videofile(target_path, fps=original_video.fps)

        


