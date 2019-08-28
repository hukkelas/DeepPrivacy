import sys
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("gpu_id", type=int)
parser.add_argument("config_path")
parser.add_argument("--extra_args", default="")

args = parser.parse_args()
filedir = os.getcwd()
#print(os.getcwd())
gpu_id = args.gpu_id

options = f"{args.config_path} {args.extra_args}"

model_name = args.config_path.split("/")[-2]
docker_container = "haakohu_{}_fid_calculation".format(model_name)
print("docker container name:", docker_container)
os.system("docker rm {}".format(docker_container))

command = "nvidia-docker run --name {} --ipc=host\
        -v /dev/log:/home/haakohu/DeepPrivacy/log -u 1174424 -v {}:/workspace -v /raid/userdata/haakohu/deep_privacy/data:/workspace/data \
           -v /raid/userdata/haakohu/deep_privacy/test_datasets/:/raid/userdata/haakohu/deep_privacy/test_datasets/\
            -v /home/haakohu/FaceDetection-DSFD:/home/haakohu/FaceDetection-DSFD \
           -e CUDA_VISIBLE_DEVICES={}  --log-opt max-size=50m\
          -it  haakohu/pytorch python -m src.metrics.fid_official.calculate_fid {}".format(
            docker_container,
            filedir,
            gpu_id,
            options
        )
#print(command)
print(options)
os.system(command)
cmd = f"nvidia-docker run --rm -it -u 1174424 -v $PWD:/workspace -e CUDA_VISIBLE_DEVICES={args.gpu_id} nvcr.io/nvidia/tensorflow:19.06-py3 python src/metrics/fid_official/calculate_fid_official.py models/{model_name}/fid_images/real/ models/{model_name}/fid_images/fake"
os.system(cmd)