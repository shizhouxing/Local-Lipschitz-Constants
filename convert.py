import torch
from torch import nn
import numpy as np
import h5py
import argparse
from models import *
from auto_LiRPA.utils import Flatten
from auto_LiRPA.patches import patches_to_matrix
try:
  from pytorch2keras import pytorch_to_keras
  from scipy.io import savemat
except ModuleNotFoundError:
  print('Warning: Unable to import pytorch_to_keras or savemat')
  print('Models cannot be converted to .h5 or .mat format')

def convert_recurjac_model(filename):
  """ Convert pre-trained models from RecurJac """
  print(f'Converting RecurJac model {filename}')
  layers = []
  activation = None
  with h5py.File(filename, 'r') as f:
    params = f['model_weights']
    for key in params:
      if key.startswith('dense'):
        layer = params[key][list(params[key].keys())[0]]
        kernel = np.array(layer['kernel:0']).transpose()
        bias = np.array(layer['bias:0'])
        layer = nn.Linear(kernel.shape[1], kernel.shape[0])
        layer.weight.data = torch.tensor(kernel)
        layer.bias.data = torch.tensor(bias)
        layers.append(layer)
        print(key, kernel.shape, bias.shape)
      elif key.startswith('relu') or key.startswith('activation'):
        activation = nn.ReLU()
        print(f'Activation {key} ReLU')
      else:
        print('Ignored', key)
  if activation is None:
    raise ValueError('Activation unknown')
  layers_ = layers
  layers = [Flatten()] + layers[:1]
  for i in range(1, len(layers_)):
    layers.append(activation)
    layers.append(layers_[i])
  model = nn.Sequential(*layers)
  print('Converted model:', model)
  return model

def conv_to_linear(conv_model, input_shape):
  x = torch.rand(input_shape)

  shapes = [x.shape]

  for module in conv_model.children():
    x = module(x)
    shapes.append(x.shape)

  # extract and convert the weights and bias to linear
  linears = []
  for i, module in enumerate(conv_model.children()):
    if isinstance(module, nn.Conv2d):
      weights = module.weight.unsqueeze(0).unsqueeze(0)
      # [1, 1, out_c, in_c, in_h, in_w]
      weights = weights.expand(*shapes[i+1][-2:], *weights.shape[2:])
      # [out_h, out_w, out_c, in_c, in_h, in_w]

      weights = weights.transpose(1,2).transpose(0,1).unsqueeze(1)

      new_weight = patches_to_matrix(
        weights, shapes[i], module.stride[0], module.padding[0])
      new_weight = new_weight.squeeze(0)
      new_weight = new_weight.view(new_weight.size(0), -1)

      new_bias = module.bias.unsqueeze(-1).expand(
          module.bias.shape[0], new_weight.shape[0]//module.bias.shape[0]
      ).reshape(-1)
      linears.append([new_weight, new_bias])
    elif isinstance(module, nn.Linear):
      linears.append([module.weight, module.bias])

  # construct the linear model
  new_modules = []
  for l in linears[:-1]:
    new_modules.append(nn.Linear(l[0].shape[1], l[0].shape[0]))
    new_modules.append(nn.ReLU())

  new_modules.append(
    nn.Linear(linears[-1][0].shape[1], linears[-1][0].shape[0]))

  linear_model = nn.Sequential(*new_modules)
  idx = 0
  for m in linear_model.children():
    if isinstance(m, nn.Linear):
      m.weight = nn.Parameter(linears[idx][0])
      m.bias = nn.Parameter(linears[idx][1])
      idx += 1

  return linear_model

def convert_to_recurjac(model, model_name, dummy_input):
  """Convert to .h5 format for recurjac."""
  print('Converting:', model)

  save_file = f'models_pretrained/{model_name}.h5'
  k_model = pytorch_to_keras(model, dummy_input)
  k_model.save(save_file, save_format='h5')

  save_file = f'models_pretrained/{model_name}.mat'
  weights = []
  for m in model.modules():
      if isinstance(m, nn.Linear):
          weights.append(np.array(m.weight.detach().cpu().numpy().tolist()))
  data = {'weights': np.array(weights, dtype=np.object)}
  savemat(save_file, data)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('model', type=str)
  parser.add_argument('saved_name', type=str)
  parser.add_argument('--data', type=str, default='synthetic')
  args = parser.parse_args()

  model = globals()[args.model]()
  model_name = args.saved_name

  if args.data == 'synthetic':
    dummy_input = torch.zeros(1, 16)
  elif args.data == 'mnist':
    dummy_input = torch.zeros(1, 1, 28, 28)
  else:
    raise ValueError(args.data)

  state_dict = torch.load(f'models_pretrained/{model_name}_ours.pth')
  model.load_state_dict(state_dict)
  if 'cnn' in model_name:
    model = conv_to_linear(model, dummy_input.shape)
  dummy_input = dummy_input.view(1, -1)

  convert_to_recurjac(model, args.saved_name, dummy_input)
