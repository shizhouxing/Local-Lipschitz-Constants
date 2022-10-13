"""Train models."""
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from models import *  # pylint: disable=wildcard-import,unused-wildcard-import
from convert import conv_to_linear
from random_dataset import RandomDataset, RandomKParameters
from torch.utils.data import TensorDataset
from datasets import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str,
                    choices=['simple', 'MNIST', 'CIFAR', 'tinyimagenet'])
parser.add_argument('--model', type=str)
parser.add_argument('--load', type=str)
parser.add_argument('--num-epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--width', type=int, default=32, help='For synthetic data')
parser.add_argument('--depth', type=int, default=3, help='For synthetic data')
parser.add_argument('--save_name', type=str)
args = parser.parse_args()

def train(model, train_loader, test_loader, dummy_input):
  model = model.cuda()
  opt = torch.optim.Adam(model.parameters(), lr = args.lr)
  for i in range(args.num_epochs):
    for (data, labels) in train_loader:
      data = data.view(-1, *dummy_input.shape[1:])
      data, labels = data.cuda(), labels.cuda()
      loss = nn.CrossEntropyLoss()(model(data), labels)
      opt.zero_grad()
      loss.backward()
      opt.step()
    acc = 0
    for (data, labels) in test_loader:
      data = data.cuda().view(-1, *dummy_input.shape[1:])
      labels = labels.cuda()
      acc += (model(data).argmax(-1) == labels).sum()
    print(f'Epoch {i}: {acc/len(test_loader.dataset):.5f}')
  print('Finished training')

def get_data_loaders():
  if args.data == 'simple':
    if not os.path.exists('data/synthetic'):
      os.makedirs('data/synthetic')
    if 'cnn' in args.model:
      input_shape = [1, 1, 8, 8]
    else:
      input_shape = [1, 16]
    n_dim = np.prod(input_shape)
    parameter = RandomKParameters(
      10000, 5*n_dim, dimension=n_dim, num_classes=10)
    dataset = RandomDataset(parameter)
    if not os.path.exists(f'data/synthetic/train_data_{n_dim}'):
      print('Generating new synthetic data')
      dataset._generate_data()
      torch.save(dataset.train_data, f'data/synthetic/train_data_{n_dim}')
      torch.save(dataset.train_labels, f'data/synthetic/train_label_{n_dim}')
      torch.save(dataset.val_data, f'data/synthetic/val_data_{n_dim}')
      torch.save(dataset.val_labels, f'data/synthetic/val_labels_{n_dim}')
    else:
      print('Using previous synthetic data')
      dataset.train_data = torch.load(f'data/synthetic/train_data_{n_dim}')
      dataset.train_labels = torch.load(f'data/synthetic/train_label_{n_dim}')
      dataset.val_data = torch.load(f'data/synthetic/val_data_{n_dim}')
      dataset.val_labels = torch.load(f'data/synthetic/val_labels_{n_dim}')
    train_data = TensorDataset(dataset.train_data, dataset.train_labels)
    test_data = TensorDataset(dataset.val_data, dataset.val_labels)
    train_data = torch.utils.data.DataLoader(
      train_data, batch_size=256, shuffle=True, pin_memory=True, num_workers=8)
    test_data = torch.utils.data.DataLoader(
      test_data, batch_size=256, pin_memory=True, num_workers=8)
    dummy_input = torch.zeros(input_shape)
  else:
    args.input_clipping = False
    args.device = 'cuda'
    dummy_input, train_data, test_data = load_data(args, args.data)

  return dummy_input, train_data, test_data

if __name__ == '__main__':
  if args.data == 'simple':
    if 'layer' in args.model: # model with varying widths
      model = globals()[args.model](width=args.width)
      args.model = f'{args.model}_{args.width}'
    elif 'width' in args.model: # model with varying depths
      model = globals()[args.model](depth=args.depth)
      args.model = f'{args.model}_{args.depth}'
    else:
      model = globals()[args.model]()
  else:
    model = globals()[args.model]()
  print(model)

  if args.load and os.path.exists(args.load):
    state_dict = torch.load(args.load)
    model.load_state_dict(state_dict['state_dict'])
  else:
    dummy_input, train_loader, test_loader = get_data_loaders()
    train(model, train_loader, test_loader, dummy_input)

  model = model.cpu()
  torch.save(model.state_dict(), f'models_pretrained/{args.model}_ours.pth')
  if 'cnn' in args.model or 'cifar' in args.model:
    model = conv_to_linear(model, dummy_input.shape)
  print('Converted CNN to MLP:', model)
  torch.save(model.state_dict(), f'models_pretrained/{args.model}_mip.pth')
