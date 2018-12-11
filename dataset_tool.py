import os
import pandas as pd
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import cv2
import multiprocessing
import glob
import torch
from torchvision.transforms.functional import to_tensor
import math

CELEBA_DATA_DIR = os.path.join("data", "celeba")
ANONYMIZED_DIR = os.path.join(CELEBA_DATA_DIR, "img_anonymized")
ANONYMIZED_LANDMARK_DIR = os.path.join(
    CELEBA_DATA_DIR, "img_anonymized_landmark")
ORIGINAL_DIR = os.path.join(CELEBA_DATA_DIR, "original2")
os.makedirs(ANONYMIZED_DIR, exist_ok=True)
os.makedirs(ANONYMIZED_LANDMARK_DIR, exist_ok=True)
os.makedirs(ORIGINAL_DIR, exist_ok=True)
images_path = os.path.join(CELEBA_DATA_DIR, "img_celeba")
assert len(os.listdir(images_path)) == 202599, "Expected 202599 images in: {}, but got: {}".format(images_path, len(os.listdir(images_path)))
bbox_file_path = os.path.join(CELEBA_DATA_DIR, "list_bbox_celeba.txt")
assert os.path.isfile(bbox_file_path), "Expected bbox list file to be in:{}".format(bbox_file_path)
landmarks_path = os.path.join(CELEBA_DATA_DIR, "list_landmarks_celeba.txt")
assert os.path.isfile(landmarks_path), "Expected landmarks file to be in : {}".format(landmarks_path)


def plot_bbox(x0, y0, width, height):
    x1, y1 = x0 + width, y0 + height
    x = [x0, x0, x1, x1, x0]
    y = [y0, y1, y1, y0, y0]
    plt.plot(x, y, "--")


def draw_landmark(obj, im, width):
    l_eye_x, l_eye_y = obj["lefteye_x"], obj["lefteye_y"]
    r_eye_x, r_eye_y = obj["righteye_x"], obj["righteye_y"]
    meye_x = l_eye_x + (r_eye_x - l_eye_x)//2
    meye_y = l_eye_y + (r_eye_y - l_eye_y)//2
    nose_x, nose_y = obj["nose_x"], obj["nose_y"]
    lmouth_x, lmouth_y = obj["leftmouth_x"], obj["leftmouth_y"]
    rmouth_x, rmouth_y = obj["rightmouth_x"], obj["rightmouth_y"]
    mmouth_x = lmouth_x + (rmouth_x - lmouth_x)//2
    mmouth_y = lmouth_y + (rmouth_y - lmouth_y)//2

    line_width = int(width*0.05)
    line_width = max(line_width, 1)

    cv2.line(im, (l_eye_x, l_eye_y), (r_eye_x, r_eye_y),
             (255, 255, 255), line_width)
    cv2.line(im, (meye_x, meye_y), (nose_x, nose_y),
             (255, 255, 255), line_width)
    cv2.line(im, (mmouth_x, mmouth_y), (nose_x, nose_y),
             (255, 255, 255), line_width)
    cv2.line(im, (lmouth_x, lmouth_y), (rmouth_x, rmouth_y),
             (255, 255, 255), line_width)


def quadratic_bounding_box(x0, y0, width, height, imshape):
    min_side = min(height, width)
    if height != width:
            side_diff = abs(height-width)
            # Want to extend the shortest side
            if min_side == height:
                # Vertical side
                height += side_diff
                if height > imshape[0]:
                    # Take full frame, and shrink width
                    height_diff = height - imshape[0]
                    y0 = 0
                    height = imshape[0]

                    side_diff = abs(height - width)
                    width -= side_diff
                    x0 += side_diff // 2
                else:
                    y0 -= side_diff // 2
                    y0 = max(0, y0)
            else:
                # Horizontal side
                width += side_diff
                if width > imshape[1]:
                    # Take full frame width, and shrink height
                    width_diff = width - imshape[1]
                    x0 = 0
                    width = imshape[1]

                    side_diff = abs(height - width)
                    height -= side_diff
                    y0 += side_diff // 2
                else:
                    x0 -= side_diff // 2
                    x0 = max(0, x0)
            # Check that bbox goes outside image
            x1 = x0 + width
            y1 = y0 + height
            if imshape[1] < x1:
                diff = x1 - imshape[1]
                x0 -= diff
            if imshape[0] < y1:
                diff = y1 - imshape[0]
                y0 -= diff
    return x0, y0, width, height


def expand_bounding_box(x0, y0, width, height, percentage, imshape):
    y0 = y0 - int(height*percentage)
    y0 = max(0, y0)
    height = height + int(height*percentage*2)
    height = min(height, imshape[0])
    x0 = x0 - int(width*percentage)
    x0 = max(0, x0)
    width = width + int(width*percentage*2)
    width = min(width, imshape[1])
    
    x0, y0, width, height = quadratic_bounding_box(x0, y0, width, height, imshape)
    return x0, y0, width, height

def anonymize_image_and_save(d, landmark):
    try:
        imname = d.image_id
        impath = os.path.join(images_path, imname)
        im = plt.imread(impath).copy()
        original_image = im.copy()
        x0, y0, width, height = d.x_1, d.y_1, d.width, d.height
        im[y0:(y0+height), x0:(x0+width)] = 0
        assert landmark.name == imname, "Expected filename: {}, but got: {}".format(
            imname, landmark.name)
        im_landmark = im.copy()
        draw_landmark(landmark, im_landmark, width)
        x0, y0, width, height = expand_bounding_box(x0, y0, width, height, 0.25, im.shape)
        im = im[y0:y0+height, x0:x0+width]
        if im.shape[0] < 128 or im.shape[1] < 128:
            return
        original_image = original_image[y0:y0+height, x0:x0+width]
        assert im.shape[0] == im.shape[1], "Expected quadratic frame. got: {}. imname: {}".format(im.shape, imname)
        im_landmark = im_landmark[y0:y0+height, x0:x0+width]
        original_impath = os.path.join(ORIGINAL_DIR, imname)
        anonymized_impath = os.path.join(ANONYMIZED_DIR, imname)
        landmark_anon_impath = os.path.join(ANONYMIZED_LANDMARK_DIR, imname)
        plt.imsave(anonymized_impath, im)
        plt.imsave(landmark_anon_impath, im_landmark)
        plt.imsave(original_impath, original_image)

        
    except Exception as e:
        print(e)
        print("Could not process image:", imname)


def extract_anonymized():
    landmarks_df = pd.read_csv(landmarks_path, skiprows=[0], delim_whitespace=True, encoding="utf-8-sig")

    bbox_df = pd.read_csv(bbox_file_path, skiprows=[0], delim_whitespace=True)


    with multiprocessing.Pool(multiprocessing.cpu_count()-2) as p:
        jobs = []
        for idx in trange(len(bbox_df)):
            d = bbox_df.iloc[idx]
            landmark = landmarks_df.iloc[idx]
            jobs.append(p.apply_async(anonymize_image_and_save, (d, landmark)))
        for j in tqdm(jobs):
            j.get()

def do_it(idx, source_dir, target_dir, files):
    imsizes = [4, 8, 16, 32, 64, 128]
    images = []
    for filepath in files:
        images.append(plt.imread(filepath))
    
    for imsize in imsizes:
        to_save = torch.zeros((len(files), 3, imsize, imsize), dtype=torch.float32)
        for i, im in enumerate(images):
                im = im[:, :, :3]
                im = cv2.resize(im, (imsize, imsize), interpolation=cv2.INTER_AREA)
                im = to_tensor(im)
                assert im.max() <= 1.0
                assert im.dtype == torch.float32
                to_save[i] = im
        td = os.path.join(target_dir, str(imsize))
        target_path = os.path.join(td, "{}.torch".format(str(idx)))
        os.makedirs(td, exist_ok=True)
        torch.save(to_save, target_path)
        del to_save


def sort_and_save_images(source_dir, target_dir):
    files = glob.glob(os.path.join(source_dir, "*.jpg"))
    files.sort(key= lambda x: os.path.basename(x))
    num_jobs = 50
    batch_size = math.ceil(len(files) / num_jobs)

    jobs = []
    with multiprocessing.Pool(16) as p:
        for i in range(num_jobs):
            f = files[i*batch_size:(i+1)*batch_size]
            j = p.apply_async(do_it, (i, source_dir, target_dir, f))
            jobs.append(j)
        for j in tqdm(jobs):
            j.get()


def dataset_to_torch():
    anon_len = len(glob.glob(os.path.join(ANONYMIZED_DIR, "*.jpg")))
    orig_len = len(glob.glob(os.path.join(ORIGINAL_DIR, "*.jpg")))
    anon_landmark_len = len(glob.glob(os.path.join(ANONYMIZED_LANDMARK_DIR, "*.jpg")))
    assert anon_len == orig_len, "Anon len: {}, orig_len: {}".format(anon_len, orig_len)
    assert anon_landmark_len == orig_len, "Anon len: {}, orig_len: {}".format(anon_landmark_len, orig_len)
    target_dir = os.path.join("data", "celeba_torch")
    # Save anonymized
    target_anonymized_dir = os.path.join(target_dir, "anonymized")
    source_dir = ANONYMIZED_DIR
    sort_and_save_images(source_dir, target_anonymized_dir)
    # Save real
    target_original_dir = os.path.join(target_dir, "original")
    source_dir = ORIGINAL_DIR
    sort_and_save_images(source_dir, target_original_dir)
    
    # Save anonymized with landmark
    target_anonymized_landmark_dir = os.path.join(target_dir, "anonymized_landmark")
    source_dir = ANONYMIZED_LANDMARK_DIR
    sort_and_save_images(source_dir, target_anonymized_landmark_dir)

if __name__ == "__main__":
    print("Started")
    #extract_anonymized()
    dataset_to_torch()
