import argparse
import os
import glob
import imageio

parser = argparse.ArgumentParser()
parser.add_argument("model_config_path")
parser.add_argument("--imsize", type=int, default=128)

args = parser.parse_args()


dirname = os.path.dirname(args.model_config_path)
image_dir = os.path.join(dirname, "generated_data", "validation")
impaths = glob.glob(os.path.join(image_dir, "*128x128.jpg"))
def get_image_id(impath):
    p = os.path.basename(impath)
    start_idx = 5
    end_idx = impath.index("_") 
    print(int(p[start_idx:end_idx]))
    return int(p[start_idx:end_idx])
impaths.sort(key=get_image_id)
impaths = impaths[::-1]
print(impaths)


images = [
    imageio.imread(p) for p in impaths
]
imageio.mimsave("test.gif", images)