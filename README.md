# DeepPrivacy
![](images/example.gif)

DeepPrivacy is a fully automatic anonymization technique for images.

This repository contains the source code for the paper [*"DeepPrivacy: A Generative Adversarial Network for Face Anonymization"*](https://arxiv.org/abs/1909.04538), published at ISVC 2019.

![](images/generated_results.png)

The DeepPrivacy GAN never sees any privacy sensitive information, ensuring a fully anonymized image. 
It utilizes bounding box annotation to identify the privacy-sensitive area, and sparse pose information to guide the network in difficult scenarios.
![](images/generated_results_annotated.png)

DeepPrivacy detects faces with state-of-the-art detection methods.
[Mask R-CNN](https://arxiv.org/abs/1703.06870) is used to generate a sparse pose information of the face, and [DSFD](https://arxiv.org/abs/1810.10220) is used to detect faces in the image.
![](images/overall_architecture.png)

## Citation
If you find this code useful, please cite the following:
```
@article{deep_privacy, 
         title={DeepPrivacy: A Generative Adversarial Network for Face Anonymization},
         url={https://arxiv.org/abs/1909.04538},
         journal={arXiv.org},
         author={Håkon Hukkelås and Rudolf Mester and Frank Lindseth},
         year={2019}
}
```
## FDF Dataset
The FDF dataset will be released at [github:hukkelas/FDF](https://github.com/hukkelas/FDF)

## Setting up your environment
Install the following: 
- Pytorch  >= 1.0.0
- Torchvision >= 0.3.0
- NVIDIA Apex (Master branch)
- Python >= 3.6

Then, install python packages:

```pip install -r docker/requirements.txt``` 

### Docker
In our experiments, we use docker as the virtual environment. 

Our docker image can be built by running:
```bash
cd docker/

docker build -t deep_privacy . 
```
Then, training can be started with:

```bash
nvidia-docker run --rm  -it -v $PWD:/app  -e CUDA_VISIBLE_DEVICES=1 deep_privacy python -m deep_privacy.train models/default/config.yml
```

## Config files
Hyperparameters and more can be set through config files, named `config.yml`.

From our paper, the following config files corresponds to our models

- `models/default/config.yml`: Default 12M parameter model with pose (Max 256 channels in convolutions.)
- `models/no_pose/config.yml`: Default 12M parameter model without pose
- `models/large/config.yml` (**BEST:**): Default 46M parameter model with pose (Max 512 channels in convolutions). If you have the compute power, we recommend to use this model.
- `models/deep_discriminator/config.yml`: Default deep discriminator model.

### Pre-trained models
For each config file, you can download pre-trained models from the following URLS:

- [`models/default/config.yml`](https://drive.google.com/open?id=1P_UO1ZSJzIUeVEkbmhc68XB3csvAyaB9)
- [`models/no_pose/config.yml`](https://drive.google.com/open?id=1hYye3ZfrILPfpRp22mzjwwMsot6RV7DJ)
- [`models/large/config.yml`](https://drive.google.com/open?id=1RXM0xIoaHARrZ87r-PFEVVOc9BjDuWc5)
- [`models/deep_discriminator/config.yml`](https://drive.google.com/drive/folders/1DZY40wh-EpoywBsNmH7nU8iNXdt-L7O3?usp=sharing)

## Automatic inference and anonymization of images
There are several scripts to perform inference

Every scripts require a path to a `config.yml` file. In these examples, we use the default model with 256 channels in the generator.

**Download Face Detector:** Before running inference, we expect that you have downloaded the DSFD face detection model, and place it to the path: `deep_privacy/detection/dsfd/weights/WIDERFace_DSFD_RES152.pth`.
This can be downloaded from the [official repository for DSFD](https://github.com/TencentYoutuResearch/FaceDetection-DSFD)
[[Google Drive Link](https://drive.google.com/file/d/1WeXlNYsM6dMP3xQQELI-4gxhwKUQxc3-/view?usp=sharing)].

### Anonymizing a single image or folder

Run
```bash
python -m deep_privacy.inference.anonymize_folder model/default/config.yml --source_path testim.jpg --target_path testim_anonymized.jpg
```

### Anonymizing Videos

Run 
```bash
python -m deep_privacy.inference.anonymize_video model/default/config.yml --source_path path/to/video.mp4 --target_path path/to/video_anonymized.mp4
```
**Note:** DeepPrivacy is a frame-by-frame method, ensuring no temporal consistency in videos.


### Anonymizing WIDER-Face Validation Datset
Run
```
python -m deep_privacy.inference.anonymize_wider models/default/config.yml --source_path path/to/Wider/face/dataset --target_path /path/to/output/folder
```
This expects the source path to include the following folders: `WIDER_val` and `wider_face_split`.


## Calculate FID scores
1. Generate real and fake images, where the last argument is the model config:
```bash
python -m deep_privacy.metrics.fid_official.calculate_fid models/default/config.yml
```

2. Calculate FID with the official tensorflow code:
```bash
python deep_privacy/metrics/fid_official/calculate_fid_official.py models/default/fid_images/real models/default/fid_images/fake
```
Where the two last arguments are the paths to real and fake images.

**NOTE:** We use nvidias tensorflow docker container to run the FID code.: [nvcr.io/nvidia/tensorflow:19.06-py3](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_19.06.html#rel_19.06)


## Training your own model

Training your own model is easy. First, download our FDF dataset, and put it under `data/fdf`.

Then run:
```bash
python -m deep_privacy.train models/default/config.yml
```


## License
All code is under MIT license, except the following:

Code under [deep_privacy/detection](deep_privacy/detection):
- DSFD is taken from [https://github.com/hukkelas/DSFD-Pytorch-Inference](https://github.com/hukkelas/DSFD-Pytorch-Inference) and follows APACHE-2.0 License
- Mask R-CNN implementation is taken from Pytorch source code at [pytorch.org](https://pytorch.org/docs/master/torchvision/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)
- FID calculation code is taken from the official tensorflow implementation: [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR)
