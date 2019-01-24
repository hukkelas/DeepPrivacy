FROM nvcr.io/nvidia/pytorch:18.08-py3

COPY  requirements.txt /workspace

RUN pip install -r requirements.txt
RUN pip install requests
RUN pip install tensorflow
RUN pip install tensorboard