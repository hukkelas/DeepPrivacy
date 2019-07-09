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
import numpy as np
CELEBA_DATA_DIR = os.path.join("data", "celeba")
TARGET_DIR = os.path.join(CELEBA_DATA_DIR, "original_cut")

os.makedirs(TARGET_DIR, exist_ok=True)
SOURCE_DIR = os.path.join(CELEBA_DATA_DIR, "img_celeba")
assert len(glob.glob(os.path.join(SOURCE_DIR, "*.jpg"))) == 202599, "Expected 202599 images in: {}, but got: {}".format(SOURCE_DIR, len(os.listdir(SOURCE_DIR)))
bbox_file_path = os.path.join(CELEBA_DATA_DIR, "list_bbox_celeba.txt")
assert os.path.isfile(bbox_file_path), "Expected bbox list file to be in:{}".format(bbox_file_path)
landmark_file_path = os.path.join(CELEBA_DATA_DIR, "list_landmarks_celeba.txt")
assert os.path.isfile(landmark_file_path)

def plot_bbox(x0, y0, width, height):
    x1, y1 = x0 + width, y0 + height
    x = [x0, x0, x1, x1, x0]
    y = [y0, y1, y1, y0, y0]
    plt.plot(x, y, "--")


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
    assert x0 >= 0, "Bounding box outside image."
    assert y0 >= 0, "Bounding box outside image."
    assert x0+width <= imshape[1], "Bounding box outside image."
    assert y0+height <= imshape[0], "Bounding box outside image."

    return x0, y0, width, height


def expand_bounding_box(x0, y0, width, height, percentage, imshape):
    x0, y0, width, height = quadratic_bounding_box(x0, y0, width, height, imshape)
    expanding_factor = int(max(height, width) * percentage)
    possible_max_expansion = [(imshape[0] - width)//2,
                              (imshape[1] - height)//2,
                              expanding_factor]
    expanding_factor = min(possible_max_expansion)
    # Expand height
    y0 = y0 - expanding_factor
    y0 = max(0, y0)

    height += expanding_factor*2
    if height > imshape[0]:
        y0 -= (imshape[0] - height)
        height = imshape[0]

    if height + y0 > imshape[0]:
        y0 -= (height + y0 - imshape[0])

    assert y0 >= 0, "Y0 is minus"
    assert height <= imshape[0], "Height is larger than image."
    # Expand width
    x0 = x0 - expanding_factor
    x0 = max(0, x0)

    width += expanding_factor*2
    if width > imshape[1]:
        x0 -= (imshape[1] - width)
        width = imshape[1]

    if width + x0 > imshape[1]:
        x0 -= (width + x0 - imshape[1])
    assert x0 + width <= imshape[1]
    assert y0 + height <= imshape[0]
    assert width == height, "HEIGHT IS NOT EQUAL WIDTH!!"
    assert x0 >= 0, "Y0 is minus"
    assert width <= imshape[1], "Height is larger than image."
    #x0, y0, width, height = quadratic_bounding_box(x0, y0, width, height, imshape)
    return x0, y0, width, height

def anonymize_image_and_save(imname, x0, y0, width, height):
    try:

        impath = os.path.join(SOURCE_DIR, imname)
        im = plt.imread(impath).copy()

        x0_exp, y0_exp, width_exp, height_exp = expand_bounding_box(x0, y0, width, height, 0.25, im.shape)
        im = im[y0_exp:y0_exp+height_exp, x0_exp:x0_exp+width_exp]

        assert width < width_exp  or (width == width_exp and x0 == x0_exp)
        assert height < height_exp or (height == height_exp and y0 == y0_exp)
        
        # Offset original bounding box to new image
        x0 -= x0_exp
        y0 -= y0_exp

        if im.shape[0] < 128 or im.shape[1] < 128:
            # Resulting image is too small. Skip it.
          return #None
        assert im.shape[0] == im.shape[1], "Expected quadratic frame. got: {}. imname: {}".format(im.shape, imname)
        target_path = os.path.join(TARGET_DIR, imname)
        plt.imsave(target_path, im)

        assert x0 >= 0, "X0 is negative"
        assert y0 >= 0, "Y0 is negative"
        assert x0 + width <= im.shape[1], "Bounding box max coord: {}, imshape: {}".format(x0+width, im.shape)
        assert y0 + height <= im.shape[0], "Bounding box max coord: {}, imshape: {}".format(y0+height, im.shape)
        return target_path, (int(x0_exp), int(y0_exp))
    except Exception as e:
        print(e)
        print("Could not process image:", imname)
        return #None


def extract_anonymized():
    bbox_df = pd.read_csv(bbox_file_path, skiprows=[0], delim_whitespace=True)
    image_start_indices = {}

    with multiprocessing.Pool(multiprocessing.cpu_count() - 2) as p:
        jobs = []
        for idx in range(len(bbox_df)):
            d = bbox_df.iloc[idx]
            jobs.append(p.apply_async(anonymize_image_and_save, (d.image_id, d.x_1, d.y_1, d.width, d.height)))
        for j in tqdm(jobs, desc="Saving original images"):
            result = j.get()
            if result is None:
              continue
            impath, saved_image_start_idx = result
            image_start_indices[impath] = saved_image_start_idx
    return image_start_indices

def save_image_batch(idx, impaths, target_dir):
    imsizes = [4, 8, 16, 32, 64, 128]
    images = []
    for impath in impaths:
        images.append(plt.imread(impath))

    for imsize in imsizes:
        to_save = np.zeros((len(impaths), imsize, imsize, 3), dtype=np.uint8)
        #to_save = torch.zeros((len(impaths), 3, imsize, imsize), dtype=torch.float32)
        for i, im in enumerate(images):
                im = im[:, :, :3]
                im = cv2.resize(im, (imsize, imsize), interpolation=cv2.INTER_AREA)
                
                #im = (im*255).astype(np.uint8)
                #im = to_tensor(im)
                assert im.dtype == np.uint8
                #assert im.dtype == torch.float32
                to_save[i] = im
        td = os.path.join(target_dir, str(imsize))
        target_path = os.path.join(td, "{}.npy".format(str(idx)))
        os.makedirs(td, exist_ok=True)
        # 
        np.save(target_path, to_save)
        #torch.save(to_save, target_path)
        del to_save
    image_heights = [im.shape[0] for im in images]
    image_widths = [im.shape[1] for im in images]        
    return image_heights, image_widths


def save_bounding_boxes(bounding_boxes, target_dir, image_height, image_width):
  target_dir = os.path.join(target_dir, "bounding_box")
  os.makedirs(target_dir, exist_ok=True)

  imsizes = [4, 8, 16, 32, 64, 128]
  for imsize in tqdm(imsizes, desc="Saving bounding boxes"):
    shrink_factor_width = (image_width.float() / imsize).view(-1, 1)
    shrink_factor_height = (image_height.float() / imsize).view(-1, 1)
    bounding_boxes_resized = bounding_boxes.float()
    bounding_boxes_resized[:, [0, 2]] /= shrink_factor_width
    bounding_boxes_resized[:, [1, 3]] /= shrink_factor_height
    bounding_boxes_resized = bounding_boxes_resized.long()

      
    assert bounding_boxes_resized.shape == bounding_boxes.shape

    target_path = os.path.join(target_dir, "{}.torch".format(imsize))
    torch.save(bounding_boxes_resized, target_path)
    del bounding_boxes_resized

def save_landmarks(landmarks, target_dir, original_image_size):
  target_dir = os.path.join(target_dir, "landmarks")
  os.makedirs(target_dir, exist_ok=True)
  #print(image_height, image_width)
  #print(landmarks)
  original_image_size = original_image_size.reshape(-1, 1)

  imsizes = [4, 8, 16, 32, 64, 128]
  for imsize in tqdm(imsizes, desc="Saving landmarks"):
    #shrink_factor_width = (image_width.float() / imsize).view(-1, 1)
    #shrink_factor_height = (image_height.float() / imsize).view(-1, 1)

    landmarks_resized = landmarks.float()
    landmarks_resized = landmarks_resized / original_image_size.float() * imsize

    assert landmarks_resized.shape == landmarks.shape

    target_path = os.path.join(target_dir, "{}.torch".format(imsize))
    torch.save(landmarks_resized, target_path)
    del landmarks_resized


def load_and_adjust_bounding_box(image_start_indices, image_paths):
    bbox_df = pd.read_csv(bbox_file_path, skiprows=[0], delim_whitespace=True)
    bounding_boxes = []
    for impath in tqdm(image_paths, desc="Loading and adjusting bounding boxes"):
        imname = os.path.basename(impath)
        image_id = int(imname.split(".")[0])
        d = bbox_df.iloc[image_id-1]
        assert d.image_id == imname, "Expected image ID to be identical: Expected: {}, was: {}".format(imname, d.image_id)
        x0, y0 = d.x_1, d.y_1
        # Adjust original bounding box
        x0_exp, y0_exp = image_start_indices[impath]
        x0 -= x0_exp
        y0 -= y0_exp
        x1 = x0 + d.width
        y1 = y0 + d.height
        bounding_boxes.append([x0, y0, x1, y1])
    bounding_boxes = np.array(bounding_boxes)
    bounding_boxes = torch.from_numpy(bounding_boxes)
    return bounding_boxes

def load_and_adjust_landmarks(image_start_indices, image_paths, landmark_df):
    landmark_df = landmark_df.set_index("imname")
    landmarks = []
    for impath in tqdm(image_paths, desc="Loading and adjusting landmarks"):
        imname = os.path.basename(impath)
        df = landmark_df.loc[imname]
        x0_exp, y0_exp = image_start_indices[impath]
        
        #df = landmark_df.iloc[image_id-1]
        #assert df.imname == imname, "Expected image ID to be identical: Expected: {}, was: {}".format(imname, df.image_id)
        to_save = df.values.squeeze().astype("float")

        assert len(to_save.shape) == 1
        to_save[range(0, len(to_save), 2)] -= x0_exp
        to_save[range(1, len(to_save), 2)] -= y0_exp
        landmarks.append(to_save)
    landmarks = np.array(landmarks)
    landmarks = torch.from_numpy(landmarks)
    return landmarks


def dataset_to_torch(image_start_indices):
    target_dir = os.path.join("data", "celeba_numpy")
    # Save real
    target_original_dir = os.path.join(target_dir, "original")
    impaths = list(image_start_indices.keys())
    landmark_df = pd.read_csv("data/celeba/celeba_keypoints.csv")
    landmark_impaths = set(landmark_df.imname.values)
    impaths = [os.path.basename(impath) for impath in impaths]
    impaths = set(impaths)
    impaths = impaths.intersection(landmark_impaths)
    #impaths = [impath for impath in impaths if os.path.basename(impath) in landmark_df.imname.values]
    impaths = list(impaths)
    impaths.sort(key=lambda x: int(x.split(".")[0]))
    impaths = [os.path.join(TARGET_DIR, x) for x in impaths]
    print("Remaining images:", len(impaths))
    num_jobs = 200
    batch_size = math.ceil(len(impaths) / num_jobs)
    jobs = []
    with multiprocessing.Pool(16) as p:
      for i in range(num_jobs):
        impath = impaths[i*batch_size: (i+1)*batch_size]
        j = p.apply_async(save_image_batch, (i, impath, target_original_dir))
        jobs.append(j)
      
      original_image_width = []
      original_image_height = []
      for j in tqdm(jobs, desc="Saving to torch"):
        image_heights, image_widths = j.get()
        original_image_height += image_heights
        original_image_width += image_widths
      original_image_height = torch.tensor(original_image_height)
      original_image_width = torch.tensor(original_image_width)
    assert original_image_height.shape == (len(impaths),)
    assert original_image_width.shape == (len(impaths),)

    bounding_boxes = load_and_adjust_bounding_box(image_start_indices,  impaths)
    save_bounding_boxes(bounding_boxes, target_dir, original_image_height, original_image_width)

    landmarks = load_and_adjust_landmarks(image_start_indices, impaths, landmark_df)
    save_landmarks(landmarks, target_dir, original_image_height)

    f = open("data/celeba_torch/idx_to_imname.csv", "w")
    f.write("imname\n")
    f.write("\n".join(impaths))
    f.close()

if __name__ == "__main__":
    image_start_indices = extract_anonymized()
    dataset_to_torch(image_start_indices)
