import torch.nn as nn
from .utils import Flatten

def simple_mlp_3layer(in_ch=1, in_dim=1, width=16):
    return nn.Sequential(
        Flatten(),
        nn.Linear(16, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, 10),
    )

def simple_mlp_width32(in_ch=1, in_dim=1, depth=2):
    modules = [Flatten()]
    shape = 16
    width = 32
    for d in range(depth-1):
        modules.append(nn.Linear(shape, width))
        shape = width
        modules.append(nn.ReLU())
    modules.append(nn.Linear(shape, 10))
    return nn.Sequential(*modules)

def simple_cnn_3layer(width=16):
    return nn.Sequential(
        nn.Conv2d(1, width, 3),
        nn.ReLU(),
        nn.Conv2d(width, width, 3),
        Flatten(),
        nn.ReLU(),
        nn.Linear(width*4*4, 10),
    )

# simple 3 layer cnn network with width 2 (converted to MLP)
def simple_cnn_3layer_2():
    return nn.Sequential(
        nn.Linear(in_features=64, out_features=72),
        nn.ReLU(),
        nn.Linear(in_features=72, out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32, out_features=10),
    )

# simple 3 layer cnn network with width 4 (converted to MLP)
def simple_cnn_3layer_4():
    return nn.Sequential(
        nn.Linear(in_features=64, out_features=144),
        nn.ReLU(),
        nn.Linear(in_features=144, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=10),
    )

# simple 3 layer cnn network with width 8 (converted to MLP)
def simple_cnn_3layer_8():
    return nn.Sequential(
        nn.Linear(in_features=64, out_features=288),
        nn.ReLU(),
        nn.Linear(in_features=288, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=10),
    )

# simple 3 layer cnn network with width 16 (converted to MLP)
def simple_cnn_3layer_16():
    return nn.Sequential(
        nn.Linear(in_features=64, out_features=576),
        nn.ReLU(),
        nn.Linear(in_features=576, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=10),
    )
