FROM  anibali/pytorch:cuda-8.0
USER root

COPY  requirements.txt /app

#RUN conda install cython
#RUN conda config --add channels conda-forge
RUN pip install -r requirements.txt
#RUN pip install scikit-learn