import torch.nn as nn
from .utils import Flatten

# simple 3 layer mlp network for MNIST dataset
def mnist_mlp_3layer(in_ch=1, in_dim=1,width = 20):
    return nn.Sequential(
        Flatten(),
        nn.Linear(784, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, 10),
    )    

# simple 4 layer cnn network with input [1, 28, 28]
def mnist_cnn_4layer(in_ch=1, in_dim=1,width=16):
    # mnist_cnn_a
    return nn.Sequential(
        nn.Conv2d(1, 8, (4,4), stride=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, (4,4), stride=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*22*22, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )

def mnist_cnn_4layer_8(in_ch=1, in_dim=1, width=16):
    return nn.Sequential(
        nn.Linear(in_features=784, out_features=5000, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=5000, out_features=3872, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=3872, out_features=100, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=100, out_features=10, bias=True)
)

def cnn_4layer_stride1_padding0(in_ch=3, in_dim=32, width=32, linear_size=256):
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=0),
        nn.ReLU(),
        Flatten(),
        nn.Linear(width * (in_dim-4)**2, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,10)
    )
    return model

def cnn_6layer_stride1_padding0(in_ch=3, in_dim=32, width=32, linear_size=256):
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=0),
        nn.ReLU(),
        Flatten(),
        nn.Linear(width * (in_dim-8)**2, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,10)
    )
    return model

def cnn_4layer_stride2_imagenet(in_ch=3, in_dim=32, width=32, linear_size=256):
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(width * (in_dim//4)**2, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 200)
    )
    return model    

def cnn_6layer_stride2_imagenet(in_ch=3, in_dim=64, width=32, linear_size=256):
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(width * (in_dim//16)**2, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 200)
    )
    return model

