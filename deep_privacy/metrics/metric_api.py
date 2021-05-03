import numpy as np
import skimage
import skimage.measure
import torch
import tqdm
import multiprocessing
try:
    from skimage.measure import compare_ssim, compare_psnr
except ImportError:
    from skimage.metrics import structural_similarity as compare_ssim
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from .perceptual_similarity import PerceptualLoss
from deep_privacy import torch_utils
from .fid_pytorch import fid as fid_api
from typing import List, Dict


def compute_metrics(
        images1: np.ndarray, images2: np.ndarray,
        metrics: List[str]) -> Dict[str, float]:
    results = {}
    METRICS = {
        "l1": l1,
        "l2": l2,
        "ssim": ssim,
        "psnr": psnr,
        "lpips": lpips,
        "fid": fid
    }
    for metric in metrics:
        func = METRICS[metric]
        results[metric] = func(images1, images2)
    return results


def check_shape(images1: np.ndarray, images2: np.ndarray):
    assert len(images1.shape) == 4
    assert images1.shape == images2.shape
    assert images1.dtype == np.float32
    assert images2.dtype == np.float32
    assert images1.max() <= 1 and images1.min() >= 0
    assert images2.max() <= 1 and images2.min() >= 0


def l2(images1: np.ndarray, images2: np.ndarray):
    check_shape(images1, images2)
    difference = (images1 - images2)**2
    rmse = difference.reshape(difference.shape[0], -1)
    rmse = rmse.mean(axis=1) ** 0.5
    return rmse.mean()


def l1(images1: np.ndarray, images2: np.ndarray):
    check_shape(images1, images2)
    difference = abs(images1 - images2)
    return difference.mean()


def ssim(images1: np.ndarray, images2: np.ndarray):
    check_shape(images1, images2)
    mean_ssim = 0
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        for img1, img2 in zip(images1, images2):
            s = pool.apply_async(
                compare_ssim, (img1, img2),
                dict(
                    data_range=1, multichannel=True, win_size=11,
                    gaussian_weights=True, sigma=1.5,
                    use_sample_covariance=False, K1=0.01 ** 2, K2=0.03 ** 2))
            jobs.append(s)
        for job in tqdm.tqdm(jobs):
            mean_ssim += job.get()
    return mean_ssim / images1.shape[0]


def psnr(images1: np.ndarray, images2: np.ndarray):
    check_shape(images1, images2)
    mean_psnr = 0
    for img1, img2 in zip(images1, images2):
        s = compare_psnr(
            img1, img2, data_range=1)
        mean_psnr += s.mean()
    return mean_psnr / images1.shape[0]


def lpips(images1: np.ndarray, images2: np.ndarray,
          batch_size: int = 64,
          metric_type: str = "net-lin",
          reduce=True):
    assert metric_type in ["net-lin", "l2", "ssim", ]

    check_shape(images1, images2)
    n_batches = int(np.ceil(images1.shape[0] / batch_size))
    model = PerceptualLoss(
        model='net-lin', net='alex', use_gpu=torch.cuda.is_available())
    distances = np.zeros((images1.shape[0]), dtype=np.float32)
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        im1 = images1[start_idx:end_idx]
        im2 = images2[start_idx:end_idx]
        im1 = torch_utils.image_to_torch(im1, normalize_img=True)
        im2 = torch_utils.image_to_torch(im2, normalize_img=True)
        with torch.no_grad():
            dists = model(im1, im2, normalize=False).cpu().numpy().squeeze()
        distances[start_idx:end_idx] = dists
    if reduce:

        return distances.mean()
    assert batch_size == 1
    return distances


def fid(images1: np.ndarray, images2: np.ndarray,
        batch_size: int = 64):
    check_shape(images1, images2)
    fid = fid_api.calculate_fid(
        images1, images2, batch_size=batch_size, dims=2048)
    return fid


def compute_all_metrics(images1: np.ndarray, images2: np.ndarray):
    metrics = {}
    metrics["L1"] = l1(images1, images2)
    metrics["L2"] = l2(images1, images2)
    metrics["PSNR"] = psnr(images1, images2)
    metrics["SSIM"] = ssim(images1, images2)
    metrics["LPIPS"] = lpips(images1, images2)
    return metrics


def print_all_metrics(images1: np.ndarray, images2: np.ndarray):
    metrics = compute_all_metrics(images1, images2)
    for m, v in metrics.items():
        print(f"{m}: {v}")


if __name__ == "__main__":
    import argparse
    import pathlib
    from deep_privacy import file_util
    parser = argparse.ArgumentParser()
    parser.add_argument("path1")
    parser.add_argument("path2")
    args = parser.parse_args()
    path1 = pathlib.Path(args.path1)
    path2 = pathlib.Path(args.path2)
    filepaths1 = file_util.find_all_files(path1)
    filepaths2 = file_util.find_matching_files(path2, filepaths1)
    images1 = file_util.read_images(filepaths1)
    images2 = file_util.read_images(filepaths2)
    print_all_metrics(images1, images2)
