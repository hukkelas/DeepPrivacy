FROM nvcr.io/nvidia/pytorch:18.09-py3

COPY  requirements.txt /workspace

RUN pip install -r requirements.txt
RUN pip install requests
RUN pip install tensorflow
RUN pip install tensorboard
RUN pip install opencv-python
#RUN pip install visdom
#RUN pip install --upgrade torchvision
#RUN pip install --upgrade https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl


RUN mkdir /pip_installs
RUN mkdir /pip_installs/apex
WORKDIR /pip_installs
#RUN pip uninstall apex -y
#RUN git clone --single-branch --branch no_wrap_optimizers https://github.com/hukkelas/apex

RUN git clone https://github.com/NVIDIA/apex.git
WORKDIR /pip_installs/apex
RUN python setup.py install 
#  --cuda_ext --cpp_ext
WORKDIR /workspace