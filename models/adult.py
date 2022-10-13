import torch.nn as nn

def adult_mlp(in_dim, width=512):
    """ Model for the Adult dataset """
    model = nn.Sequential(
        nn.Linear(in_dim, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, 2))
    return model
