# DeepPrivacy

## Requirements
- Pytorch  1.0.0
- NVIDIA Apex (Master branch)
- Python >= 3.6
- Apex for pytorch
- NVIDIA GPU

Install dependencies 

```pip install -r requirements.txt``` 

## Pre-trained model
Is included with the submission files. Unzip it to models/large_v2/checkpoints

## Get started 

Start by setting hyperparameters in a config file. These are normally saved under `models/` directory.

To start training a model:

```bash
python train.py models/default/config.yml
```

It will automatically look for previous training checkpoints in `models/default/checkpoints`. 

Additional arguments can be found with:

```python
python train.py -h 
```

Launch tensorboard

```bash
tensorboard --logdir summaries/
```

## Automatic inference and anonymization of images

Run
```bash
python -m deep_privacy.inference.batch_infer [config_path] --source_path /path/to/source/directory --target_path /path/to/taget/directory
```
By default it will look for images in the folder `test_examples`, and images will be saved to the same path as the config file.




## Get started (Docker)

1. Build docker image file ( Done from same folder as `Dockerfile`) 

```bash
docker build -t pytorch-gpu-ext . 
```

2. Run training with docker (Launch in same folder as `train.py`)
```bash
nvidia-docker run --rm  -it -v $PWD:/app  -e CUDA_VISIBLE_DEVICES=1 pytorch-gpu-ext python train.py models/large_v2/config.yml
```
