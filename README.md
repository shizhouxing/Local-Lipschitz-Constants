# Efficiently Computing Local Lipschitz Constants of Neural Networks via Bound Propagation

Code for paper:

[Efficiently Computing Local Lipschitz Constants of Neural Networks via Bound Propagation](https://arxiv.org/abs/2210.07394),
by Zhouxing Shi, Yihan Wang, Huan Zhang, Zico Kolter and Cho-Jui Hsieh. In NeurIPS 2022.

The core implementation of this paper is now a part of [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA).
See the [example](https://github.com/Verified-Intelligence/auto_LiRPA/blob/master/examples/vision/jacobian.py) about bounding Jacobian, Jacobian-vector product, and Linf local Lipschitz constants.

For reproducing our results, please install [auto\_LiRPA version 0.3.1](https://github.com/Verified-Intelligence/auto_LiRPA/tree/d2592c13198e0eb536967186758c264604b59539) which was released in November, 2022. 
We are working on a more general and flexible support for Jacobian in auto\_LiRPA.

## Dependencies

Python 3.7+ and PyTorch 1.11+ are recommended.

Install other Python libraries:
```bash
pip install -r requirements.txt
```

## Train Models

We first need to train models which will be saved to `models_pretrained/`
for analyzing local Lipschitz constants:
```bash
python train_model.py --data DATA --model MODEL
```

`DATA` can be chosen from `simple`, `MNIST`, `CIFAR`, and `tinyimagenet`.
For `tinyimagenet`, data need to be downloaded first with:
```bash
cd data/tinyImageNet
bash tinyimagenet_download.sh
```

`MODEL` can be chosen from models available under `models/`.
Some models in `models/simple.py` have arguments `width` and `depth` for experiments
with varying width or depth, and they can be set by `--width WIDTH` or `--depth DEPTH` respectively.

Other options include `--num-epochs` and `--lr` for setting number of epochs and learning rates.

## Compute Local Lipschitz Constants

To compute local Lipschitz constants by our method, we run:
```bash
python main.py --data DATA --model MODEL --load PATH_TO_MODEL_FILE --eps EPS
```
`PATH_TO_MODEL_FILE` is the path to the checkpoint of the pretrained model, and `EPS`
is the radius of input domain. For models from `models/simple.py`, their width and depth
can be specified by `--model-params width=WIDTH` or `--model-params depth=DEPTH`.

By default, BaB is not used. To enable Branch-and-Bound (BaB), add `--bab`, and the time
budget can be set by `--timeout TIMEOUT`. Batch size of BaB can be set by `--batch-size BATCH-SIZE`
to fit it into the GPU memory.
