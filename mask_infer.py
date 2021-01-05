import pathlib
import numpy as np
import typing
import cv2
import time
from deep_privacy import utils, file_util
from deep_privacy.config import Config,  default_parser
from deep_privacy.inference import infer, inpaint_inference
import torch

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = True


def get_paths(args):
    image_path = pathlib.Path(args.image_path)
    mask_path = args.mask_path
    single_file = not image_path.is_dir()

    if single_file:
        image_paths = [image_path]
        if mask_path is None:
            filename = image_path.stem + "_mask" + image_path.suffix
            mask_path = image_path.parent.joinpath(filename)
            assert mask_path.is_file(),\
                f"Did not find mask at location: {mask_path}"
        else:
            mask_path = pathlib.Path(mask_path)
        mask_paths = [mask_path]
    else:
        image_paths = file_util.find_all_files(image_path)
        if mask_path is None:
            mask_path = image_path.parent.joinpath("masks")
        mask_paths = file_util.find_matching_files(
            mask_path,
            image_paths)

    target_path = args.target_path
    if target_path is None:
        if single_file:
            filename = image_path.stem + "_result" + image_path.suffix
            target_path = image_path.parent.joinpath(filename)
            target_paths = [target_path]
        else:
            target_path = image_path.parent.joinpath("result")
            target_path.mkdir(exist_ok=True, parents=True)
            target_paths = []
            for impath in image_paths:
                target_paths.append(
                    target_path.joinpath(impath.name)
                )
    else:
        target_paths = [pathlib.Path(target_path)]
    return image_paths, mask_paths, target_paths


def is_same_shape(images: typing.List[np.ndarray]):
    shape1 = images[0].shape
    for im in images:
        if im.shape != shape1:
            return False
    return True



if __name__ == "__main__":
    parser = default_parser()
    parser.add_argument(
        "-i", "--image_path", default="data/validation_datasets/celebA-HQ",
    )
    parser.add_argument(
        "-m", "--mask_path", default=None,
        help="Path to mask dir/file. Sets the default to _mask file or image_path/../mask"
    )
    parser.add_argument(
        "-t", "--target_path", default=None
    )
    parser.add_argument(
        "--step", default=None,
        type=int,
        help="Load a specific step from the validation checkpoint dir"
    )
    parser.add_argument(
        "--batch_size", default=None,
        type=int,
        help="Batch size for generator"
    )
    args = parser.parse_args()
    cfg = Config.fromfile(args.config_path)
    generator = infer.load_model_from_checkpoint(
        cfg,
        args.step
    )
    image_paths, mask_paths, target_paths = get_paths(args)
    assert len(image_paths) > 0, f"found no images in {args.image_path}"

    images, masks = file_util.read_mask_images(
        image_paths, mask_paths, generator.current_imsize)
    start = time.time()
    inpainted_images, inputs = inpaint_inference.inpaint_images(
        images, masks, generator)
    tot_time = time.time() - start
    avg_time = tot_time / inpainted_images.shape[0]
    fps = inpainted_images.shape[0] / tot_time
    print(f"Inpainted {inpainted_images.shape[0]} in {tot_time:.2f} seconds. FPS: {fps}, Average time: {avg_time}")
    for (image, masked_out), target_path in zip(zip(inpainted_images, inputs), target_paths):
        image = image
        cv2.imwrite(
            str(target_path),
            image[:, :, ::-1])
#        print("Saving to:", target_path)
        input_im_path = target_path.parent.parent.joinpath("masked_out")
        input_im_path.mkdir(exist_ok=True)
        input_im_path = input_im_path.joinpath(target_path.name)
        cv2.imwrite(str(input_im_path), (masked_out*255)[:, :, ::-1].astype(np.uint8))