# Reproducing GCPR results


## Models (Table 1)

The models in table 1 can be trained with the configs in configs/gcpr/fdf and configs/gcpr/places



```
python train.py config_path.py
```

For Celeb-A and Places2 final results, use the following configs:
- CelebA: `configs/gcpr/celebA256.py`
- Places2: `configs/places/places256.py`

## Image inpainting inference
Place the model you want to test in the output directory corresponding to the config file.

For example, if you want to test `configs/gcpr/places/places256.py` model do:

1. Find output dir: `python -m deep_privacy.config.base configs/gcpr/places/places256.py
2. That should print `outputs/gcpr/places/places256/`
2. Copy the checkpoint to the "checkpoints" folder in the output dir. `outputs/gcpr/places/places256/checkpoints`
3. Run mask infer with the config.

```
python3 mask_infer.py -c config_path.py -i image_file.png -m mask_path -t target_path
```

### Tasble 1 Models and pretrained checkpoints
The link to the model checkpoints will be deprecated in a couple of months. If they do not work, you can download the models from here: [https://bird.unit.no/resources/91b243f2-0807-4e93-881e-ff1717dabc48/content](https://bird.unit.no/resources/91b243f2-0807-4e93-881e-ff1717dabc48/content)

| Configs | FDF | Places2 |
|---------|--------|-----|
| B|[[config]](configs/gcpr/fdf/gcpr_B.py)[[checkpoint]](http://folk.ntnu.no/haakohu/checkpoints/GCPR/fdf_B.ckpt)|[[config]](configs/gcpr/places/places_A.py)[[checkpoint]](http://folk.ntnu.no/haakohu/checkpoints/GCPR/places_B.ckpt)|
| C|[[config]](configs/gcpr/fdf/gcpr_C.py)[[checkpoint]](http://folk.ntnu.no/haakohu/checkpoints/GCPR/fdf_C.ckpt)|--|
| D|[[config]](configs/gcpr/fdf/gcpr_D.py)[[checkpoint]](http://folk.ntnu.no/haakohu/checkpoints/GCPR/fdf_D.ckpt)|[[config]](configs/gcpr/places/places_D.py)[[checkpoint]](http://folk.ntnu.no/haakohu/checkpoints/GCPR/places_D.ckpt)|
| E|[[config]](configs/gcpr/fdf/gcpr_E.py)[[checkpoint]](http://folk.ntnu.no/haakohu/checkpoints/GCPR/fdf_E.ckpt)|[[config]](configs/gcpr/places/places_E.py)[[checkpoint]](http://folk.ntnu.no/haakohu/checkpoints/GCPR/places_E.ckpt)|

## Environment
All models are trained with the following docker file on V100-32GB
```
FROM nvcr.io/nvidia/pytorch:20.08-py3
RUN useradd -ms /bin/bash -u 1174424 haakohu && \
    mkdir -p /home/haakohu &&\
    chown -R 1174424 /home/haakohu

WORKDIR /home/haakohu
COPY  requirements.txt /home/haakohu

RUN pip install -r requirements.txt
RUN pip install 'git+https://github.com/facebookresearch/fvcore'
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Install Detectron2
RUN git clone https://github.com/facebookresearch/detectron2 /detectron2_repo
ENV FORCE_CUDA="1"
RUN pip install -e /detectron2_repo
# Add densepose to python path
ENV PYTHONPATH="${PYTHONPATH}:/detectron2_repo/projects/DensePose"


RUN apt-get update
RUN apt install ffmpeg -y
RUN pip install addict albumentations face_detection
RUN pip install tensorflow-gpu
RUN pip install wandb
```


A pip list shows the following environment:
```
absl-py==0.9.0
addict==2.3.0
alabaster==0.7.12
albumentations==0.4.6
apex==0.1
argon2-cffi==20.1.0
ascii-graph==1.5.1
astunparse==1.6.3
attrs==19.3.0
audioread==2.1.8
autopep8==1.5.4
Babel==2.8.0
backcall==0.2.0
beautifulsoup4==4.9.1
bleach==3.1.5
blis==0.4.1
boto3==1.14.45
botocore==1.17.45
brotlipy==0.7.0
cachetools==4.1.1
catalogue==1.0.0
certifi==2020.6.20
cffi==1.14.0
chardet==3.0.4
click==7.1.2
cloudpickle==1.6.0
codecov==2.1.8
conda==4.8.3
conda-build==3.18.11
conda-package-handling==1.7.0
configparser==5.0.0
coverage==5.2.1
cryptography==2.9.2
cxxfilt==0.2.2
cycler==0.10.0
cymem==2.0.3
Cython==0.28.4
DataProperty==0.50.0
decorator==4.4.2
defusedxml==0.6.0
-e git+https://github.com/facebookresearch/detectron2@cb7e86f821f0bedb3da8f1e0a527dd821d8e5f7e#egg=detectron2
DLLogger @ git+https://github.com/NVIDIA/dllogger@26a0f8f1958de2c0c460925ff6102a4d2486d6cc
docker-pycreds==0.4.0
docutils==0.15.2
entrypoints==0.3
face-detection==0.2.1
filelock==3.0.12
flake8==3.7.9
Flask==1.1.2
future==0.18.2
fvcore @ git+https://github.com/facebookresearch/fvcore@1411919d5d351f5c0de4c7becbb5e7bb6c4b5b89
gast==0.3.3
gitdb==4.0.5
GitPython==3.1.8
glob2==0.7
google-auth==1.21.3
google-auth-oauthlib==0.4.1
google-pasta==0.2.0
grpcio==1.31.0
h5py==2.10.0
html2text==2020.1.16
hypothesis==4.50.8
idna==2.9
imageio==2.9.0
imageio-ffmpeg==0.4.2
imagesize==1.2.0
imgaug==0.4.0
importlib-metadata @ file:///tmp/build/80754af9/importlib-metadata_1593446433964/work
inflect==4.1.0
iniconfig==1.0.1
ipdb==0.13.3
ipykernel==5.3.4
ipython @ file:///tmp/build/80754af9/ipython_1593447367857/work
ipython-genutils==0.2.0
itsdangerous==1.1.0
jedi @ file:///tmp/build/80754af9/jedi_1592841914522/work
Jinja2==2.11.2
jmespath==0.10.0
joblib==0.16.0
json5==0.9.5
jsonschema==3.0.2
jupyter-client==6.1.6
jupyter-core==4.6.3
jupyter-tensorboard==0.2.0
jupyterlab==1.2.14
jupyterlab-server==1.2.0
jupytext==1.5.2
Keras-Preprocessing==1.1.2
kiwisolver==1.2.0
libarchive-c==2.9
librosa==0.6.3
llvmlite==0.28.0
lmdb==0.99
Markdown==3.2.2
MarkupSafe==1.1.1
maskrcnn-benchmark @ file:///opt/pytorch/examples/maskrcnn/pytorch
matplotlib==3.3.1
mbstrdecoder==1.0.0
mccabe==0.6.1
mistune==0.8.4
mlperf-compliance==0.0.10
mock==4.0.2
more-itertools==8.4.0
moviepy==1.0.3
msgfy==0.1.0
murmurhash==1.0.2
nbconvert==5.6.1
nbformat==5.0.7
networkx==2.0
nltk==3.5
notebook==6.1.3
numba==0.43.1
numpy==1.18.5
nvidia-dali-cuda110==0.24.0
oauthlib==3.1.0
onnx @ file:///opt/pytorch/pytorch/third_party/onnx
onnxruntime==1.4.0
opencv-python==4.4.0.44
opt-einsum==3.3.0
packaging==20.4
pandas==0.24.2
pandocfilters==1.4.2
parso==0.7.0
pathtools==0.1.2
pathvalidate==2.3.0
pexpect==4.8.0
pickleshare==0.7.5
Pillow==7.2.0
Pillow-SIMD @ file:///tmp/pillow-simd
pkginfo==1.5.0.1
plac @ file:///tmp/build/80754af9/plac_1594261902054/work
pluggy==0.13.1
portalocker==2.0.0
preshed==3.0.2
proglog==0.1.9
progressbar==2.5
prometheus-client==0.8.0
promise==2.3
prompt-toolkit==3.0.5
protobuf==3.13.0
psutil==5.7.0
ptyprocess==0.6.0
py==1.9.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
pybind11==2.5.0
pycocotools==2.0.2
pycodestyle==2.6.0
pycosat==0.6.3
pycparser==2.20
pydot==1.4.1
pyflakes==2.1.1
Pygments==2.6.1
pynvml==8.0.4
pyOpenSSL @ file:///tmp/build/80754af9/pyopenssl_1594392929924/work
pyparsing==2.4.7
pyprof @ file:///opt/pytorch/pyprof
pyrsistent==0.16.0
PySocks==1.7.1
pytablewriter==0.47.0
pytest==6.0.1
pytest-cov==2.10.1
pytest-pythonpath==0.7.3
python-dateutil==2.8.1
python-hostlist==1.20
python-nvd3==0.15.0
python-slugify==4.0.1
pytorch-transformers==1.1.0
pytz==2020.1
PyWavelets==1.1.1
PyYAML==5.3.1
pyzmq==19.0.2
regex==2020.7.14
requests @ file:///tmp/build/80754af9/requests_1592841827918/work
requests-oauthlib==1.3.0
resampy==0.2.2
revtok @ git+git://github.com/jekbradbury/revtok.git@f1998b72a941d1e5f9578a66dc1c20b01913caab
rsa==4.6
ruamel-yaml==0.15.87
s3transfer==0.3.3
sacrebleu==1.2.10
sacremoses==0.0.35
scikit-image==0.15.0
scikit-learn==0.23.2
scipy @ file:///tmp/build/80754af9/scipy_1592930497347/work
Send2Trash==1.5.0
sentencepiece==0.1.91
sentry-sdk==0.17.8
Shapely==1.7.1
shortuuid==1.0.1
six==1.15.0
smmap==3.0.4
snowballstemmer==2.0.0
SoundFile==0.10.3.post1
soupsieve==2.0.1
sox==1.4.0
spacy @ file:///tmp/build/80754af9/spacy_1594303279343/work
Sphinx==3.2.1
sphinx-rtd-theme==0.5.0
sphinxcontrib-applehelp==1.0.2
sphinxcontrib-devhelp==1.0.2
sphinxcontrib-htmlhelp==1.0.3
sphinxcontrib-jsmath==1.0.1
sphinxcontrib-qthelp==1.0.3
sphinxcontrib-serializinghtml==1.1.4
srsly==1.0.2
SSD @ file:///opt/pytorch/examples/ssd
subprocess32==3.5.4
subword-nmt @ git+git://github.com/rsennrich/subword-nmt.git@48ba99e657591c329e0003f0c6e32e493fa959ef
tabledata==1.1.3
tabulate==0.8.7
tensorboard==2.3.0
tensorboard-plugin-dlprof @ file:///nvidia/opt/tensorboard_install/tensorboard_plugin_dlprof-0.6-py3-none-any.whl
tensorboard-plugin-wit==1.7.0
tensorflow-estimator==2.3.0
tensorflow-gpu==2.3.1
tensorrt==7.1.3.4
termcolor==1.1.0
terminado==0.8.3
testpath==0.4.4
text-unidecode==1.3
tflib==0.1.0
thinc @ file:///tmp/build/80754af9/thinc_1594251955397/work
threadpoolctl==2.1.0
toml==0.10.1
torch @ file:///opt/pytorch/pytorch
torchtext @ file:///opt/pytorch/text
torchvision @ file:///opt/pytorch/vision
tornado==6.0.4
tqdm==4.31.1
traitlets==4.3.3
typepy==1.1.1
typing==3.7.4.3
typing-extensions==3.7.4.2
Unidecode==1.1.1
urllib3==1.25.9
wandb==0.10.4
wasabi==0.6.0
watchdog==0.10.3
wcwidth @ file:///tmp/build/80754af9/wcwidth_1593447189090/work
webencodings==0.5.1
Werkzeug==1.0.1
wrapt==1.12.1
yacs==0.1.8
zipp==3.1.0

```