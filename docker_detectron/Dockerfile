# Use Caffe2 image as parent image
FROM nvcr.io/nvidia/pytorch:18.12-py3


# Clone the Detectron repository
RUN git clone https://github.com/facebookresearch/detectron /detectron

# Install Python dependencies
RUN pip install -r /detectron/requirements.txt

RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
WORKDIR /cocoapi/PythonAPI
RUN make install

# Go to Detectron root
WORKDIR /detectron

# Set up Python modules
RUN make

COPY  requirements.txt /workspace
WORKDIR /workspace
RUN pip install -r requirements.txt
RUN pip install flickrapi

RUN mkdir /pip_installs
RUN mkdir /pip_installs/apex
WORKDIR /pip_installs
RUN pip uninstall apex -y
RUN git clone https://github.com/NVIDIA/apex.git

#RUN git clone https://github.com/NVIDIA/apex.git
WORKDIR /pip_installs/apex
RUN python setup.py install --cuda_ext --cpp_ext
WORKDIR /workspace