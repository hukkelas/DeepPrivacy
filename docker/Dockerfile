FROM nvcr.io/nvidia/pytorch:19.06-py3

COPY  requirements.txt /workspace

RUN pip install -r requirements.txt
RUN pip install requests
RUN pip install tensorflow --ignore-installed
RUN pip install tensorboard
RUN pip install opencv-python


RUN mkdir /pip_installs
RUN mkdir /pip_installs/apex
WORKDIR /pip_installs
RUN pip uninstall -y apex
RUN git clone https://github.com/NVIDIA/apex.git
WORKDIR /pip_installs/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
WORKDIR /workspace