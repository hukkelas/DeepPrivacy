# S³FD: Single Shot Scale-invariant Face Detector
A PyTorch Implementation of Single Shot Scale-invariant Face Detector.

## Eval
```
python wider_eval_pytorch.py

cd eval/eval_tools_old-version
octave wider_eval_pytorch.m
```
## Model
[s3fd_convert.7z](https://github.com/clcarwin/SFD_pytorch/releases/tag/v0.1)

## Test
```
python test.py --model data/s3fd_convert.pth --path data/test01.jpg
```
![output](data/test01_output.png)

# References
[SFD](https://github.com/sfzhang15/SFD)
