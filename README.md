# DeepPrivacy

## Requirements
- Pytorch  1.0.0
- Apex
- Python >= 3.6
- Apex for pytorch
- NVIDIA GPU

Install dependencies 

```pip install -r requirements.txt``` 

## Pre-trained model

Download from [https://drive.google.com/open?id=16NeOjlEaJnuH_HWHvFRyGHSGMflO3vvU](Google Drive)

## Dataset download and pre-processing
1. The dataset is rather large (83GB). Contact either Håkon Hukkelås (hakon.hukkelas@ntnu.no) or Frank Lindseth (frankl@ntnu.no) to retrieve the dataset


## Get started 

To start training a model:

```bash
python train.py
```

Hyperparameters etc can be set with arguments. For a full list of arguments run:

```python
python train.py -h 
```



To continue training on a previous model

```bash
python train.py --model model_name
```

Launch tensorboard

```bash
tensorboard --logdir summaries/
```

Run scripts to perform experiments on a trained model from the scripts/ folder. E.g: 

```bash
python -m scripts.automatic_metric_test
```


## Automatic inference and anonymization of images

Run
```bash
python -m detectron_scripts.real_image_eval [model_name]
```
And it will look for images in the folder `test_examples/real_image/test/source` and output images will be put to: `test_examples/real_image/test/out`

**NOTE** This requires detectron to be installed. A pre-defined dockerfile is given in detectron_docker to achieve this.


## Get started (Docker)

1. Build docker image file ( Done from same folder as `Dockerfile`) 

```bash
docker build -t pytorch-gpu-ext . 
```

2. Run training with docker (Launch in same folder as `train.py`)
```bash
nvidia-docker run --rm  -it -v $PWD:/app  -e CUDA_VISIBLE_DEVICES=1 pytorch-gpu-ext python train.py 
```

`CUDA_VISIBLE_DEVIES` sets what GPU to use
`coolImageName` is the docker image to use (created in step 1) 
