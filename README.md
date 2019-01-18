# DeepPrivacy

## Requirements
- Pytorch  0.4.1
- Python >= 3.6

Install dependencies 

```pip install -r requirements.txt``` 

## Download dataset and pre-process it
1. Download the celebA dataset from [http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
2. Place img_celeba in `data/celeba/img_celeba`  and the `list_bbox_celeba.txt` in the folder `data/celeba`
3. Run `python dataset_tool.py` (PS: This can take several hours on a low-end computer.)
4. The dataset should be saved to  `data/celeba_torch` in .torch files



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

On `telenor001` server the image `pytorch-gpu-ext` can be used as well.
