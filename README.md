# Efficiently Computing Local Lipschitz Constants of Neural Networks via Bound Propagation

Code for paper:

Efficiently Computing Local Lipschitz Constants of Neural Networks via Bound Propagation,
by Zhouxing Shi, Yihan Wang, Huan Zhang, Zico Kolter and Cho-Jui Hsieh.
To appear in NeurIPS 2022.

The code is partly based on [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA)
and [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN).

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

## Code Under Development

This code is still under development. We will gradually integrate part of the code
into [auto\_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA). We will
also release some pre-trained models used in the experiments.
