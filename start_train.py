import sys
import os
filedir = os.path.dirname(os.path.abspath(__file__))
gpu_id = sys.argv[1]
try:
    print("Starting training on:", int(gpu_id))
    num_gpus = 1
except Exception:
    print("Starting training on:", [int(x) for x in gpu_id.split(",")])
    num_gpus = len(gpu_id.split(","))

options = " ".join(sys.argv[2:])

model_name = sys.argv[2].split("/")[-2]
#model_index = sys.argv.index("--model")

#model_name = sys.argv[model_index+1]
docker_container = "haakohu_{}".format(model_name)
print("docker container name:", docker_container)
os.system("docker rm {}".format(docker_container))

distributed_command = "" #if num_gpus <= 1 else "-m torch.distributed.launch --nproc_per_node {}".format(num_gpus)
command = "nvidia-docker run --name {} --ipc=host\
        -v /dev/log:/home/haakohu/DeepPrivacy/log -u 1174424 -v {}:/workspace -v /raid/userdata/haakohu/deep_privacy/data:/workspace/data \
           -e CUDA_VISIBLE_DEVICES={}  --log-opt max-size=50m\
          -it  haakohu/pytorch python -m src.train {}".format(
            docker_container,
            filedir,
            gpu_id,
            options
        )
#print(command)
print(options)
os.system(command)
