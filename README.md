# DeepPrivacy


## Requirements
Pytorch >= 0.4
Python >= 3.6

## Get started (Local machine)

1. Install dependencies: 

```python
pip install -r requirements.txt
```
2. Train network

```python
python train.py
```

## Get started (Docker)

1. Build docker image file ( Done from same folder as `Dockerfile`) 

```bash
docker build -t coolImageName . 
```

2. Run training with docker (Launch in same folder as `train.py`)
```bash
nvidia-docker run --rm  -it -v $PWD:/app  -e CUDA_VISIBLE_DEVICES=1 coolImageName python train.py 
```

`CUDA_VISIBLE_DEVIES` sets what GPU to use
`coolImageName` is the docker image to use (created in step 1) 

On `telenor001` server the image `pytorch-gpu-ext` can be used as well.



# Milestones

1. Replicate Progressive - GAN
2. Train with context information

# TODO:

1. Implement CelebA
2. Implement upscaling of GAN
3. Train on generated data
