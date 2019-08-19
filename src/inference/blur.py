import tqdm
import cv2
from src.inference.anonymizer import Anonymizer
from .batch_infer import pre_process_faces
from .batch_infer import batch_post_process
from src.visualization import utils as vis_utils


def is_height_larger(bbox, image_shape, max_face_size):
    x0, y0, x1, y1 = bbox
    face_size = (y1 - y0) / image_shape[0]
    return face_size >= max_face_size

class PixelationAnonymizer(Anonymizer):


    def __init__(self, pixelation_size, save_debug=False):
        super().__init__(save_debug=save_debug)
        self.pixelation_size = pixelation_size
        print("Pixelation initialize!")
    
    def anonymize_images(self, images, im_bboxes, max_face_size=1.0):
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
                face = cv2.resize(face, (self.pixelation_size,
                                         self.pixelation_size))
                face = cv2.resize(face, (x1 - x0, y1 - y0))
                anonymized_image[y0:y1, x0:x1] = face
            
            if self.save_debug:
                anonymized_image = vis_utils.draw_faces(anonymized_image,
                                                        bboxes)

            anonymized_images.append(anonymized_image)
        return anonymized_images


if __name__ == "__main__":
    a = PixelationAnonymizer(4, False)
    a.anonymize_video("test_examples/video/NAPLab_Video.mp4",
                      "test_examples/video/NAPLab_Video_anonymized.mp4",
                      (4*60+30)*25, None)