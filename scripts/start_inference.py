import sys
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("gpu_id", type=int)
parser.add_argument("config_path")
parser.add_argument("inference_type", default="standard")
parser.add_argument("--extra_args", default="")

args = parser.parse_args()
filedir = os.path.dirname(os.path.abspath(__file__))
gpu_id = args.gpu_id
print(args)
options = f"{args.config_path} {args.extra_args}"

model_name = args.config_path.split("/")[-2]
docker_container = "haakohu_{}_inference_{}".format(model_name, gpu_id)
print("docker container name:", docker_container)
os.system("docker rm {}".format(docker_container))

if args.inference_type == "standard":
  python_file = "batch_infer"
elif args.inference_type == "wider":
  python_file = "infer_wider"
elif args.inference_type == "video":
  python_file = "deep_privacy_anonymizer"
elif args.inference_type == "blur":
  python_file = "blur"
else:
  raise AttributeError

command = "nvidia-docker run --name {} --ipc=host\
        -v /dev/log:/home/haakohu/DeepPrivacy/log -u 1174424 -v {}:/workspace -v /raid/userdata/haakohu/deep_privacy/data:/workspace/data \
           -v /raid/userdata/haakohu/deep_privacy/test_datasets/:/raid/userdata/haakohu/deep_privacy/test_datasets/\
            -v /home/haakohu/FaceDetection-DSFD:/home/haakohu/FaceDetection-DSFD \
           -e CUDA_VISIBLE_DEVICES={}  --log-opt max-size=50m\
          -it  haakohu/pytorch python -m deep_privacy.inference.{} {}".format(
            docker_container,
            filedir,
            gpu_id,
            python_file,
            options
        )
#print(command)
print(command)

print(options)
os.system(command)
