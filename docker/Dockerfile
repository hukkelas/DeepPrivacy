FROM nvcr.io/nvidia/pytorch:22.08-py3
ENV DEBIAN_FRONTEND=nonintercative
ENV FORCE_CUDA="1"
ENV TORCH_HOME=/workspace

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 ffmpeg -y
RUN pip install  cython tqdm moviepy opencv-python pyyaml scikit-image albumentations addict 
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git@0703e08a5f589f7503a3fbfce41309c80204eec8'
WORKDIR /workspace
RUN mkdir /pytorch_models
RUN pip install git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git
RUN pip install opencv-python==4.5.5.64