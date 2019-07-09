FROM nvcr.io/nvidia/pytorch:19.01-py3
RUN mkdir /pip_installs

RUN pip uninstall -y torch torchvision

RUN pip install --upgrade setuptools
RUN pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
RUN pip install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl
COPY  requirements.txt /workspace

RUN pip install -r requirements.txt
RUN pip install requests
RUN pip install tensorboard
RUN pip install opencv-python

RUN mkdir /pip_installs/apex
WORKDIR /pip_installs

RUN pip uninstall -y apex
RUN git clone https://github.com/NVIDIA/apex.git
RUN pip install --upgrade numpy
WORKDIR /pip_installs/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
WORKDIR /workspace

RUN mkdir /pytorch_models
ENV TORCH_HOME=/workspace

