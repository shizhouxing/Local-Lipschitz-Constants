import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset, Dataset
import os
import json
import pandas

class Adult(Dataset):
  """https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"""

  url_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data' # pylint: disable=line-too-long
  url_test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test' # pylint: disable=line-too-long
  path_train = 'adult/adult.data'
  path_test = 'adult/adult.test'
  path_metadata = 'adult/metadata.json'

  def __init__(self, root, train):
    super().__init__()
    if not os.path.exists(os.path.join(root, 'adult')):
      os.makedirs(os.path.join(root, 'adult'))
    self.path_train = os.path.join(root, Adult.path_train)
    self.path_test = os.path.join(root, Adult.path_test)
    if not os.path.exists(self.path_train):
      os.system(f'wget {Adult.url_train} -O {self.path_train}')
    if not os.path.exists(self.path_test):
      os.system(f'wget {Adult.url_test} -O {self.path_test}')
    self.path_metadata = os.path.join(root, Adult.path_metadata)
    if not os.path.exists(self.path_metadata):
      self.build_metadata()
    with open(self.path_metadata) as file:
      self.metadata = json.loads(file.read())
    print(f'Metadata for features: {self.metadata}')
    self.num_features = len(self.metadata)
    self.train = train
    if train:
      self.load_data(self.path_train)
    else:
      self.load_data(self.path_test)

  def build_metadata(self):
    csv = pandas.read_csv(self.path_train, header=None)
    num_features = len(csv.keys()) - 1
    metadata = []
    for i in range(num_features):
      discrete = isinstance(csv[i][0], str)
      if discrete:
        metadata.append({
          'type': 'discrete',
          'values': list(set(csv[i]))
        })
      else:
        metadata.append({
          'type': 'continuous',
          'min': float(csv[i].min()),
          'max': float(csv[i].max()),
          'mean': float(csv[i].mean()),
          'std': float(csv[i].std()),
        })
    with open(self.path_metadata, 'w') as file:
      file.write(json.dumps(metadata))

  def load_data(self, path):
    csv = pandas.read_csv(path, header=None, skiprows=[] if self.train else [0])
    self.num_features = 0
    for i, item in enumerate(self.metadata):
      if item['type'] == 'continuous':
        print(f'continuous feature {i} -> {self.num_features}')
        self.num_features += 1
      else:
        self.num_features += len(item['values'])
    self.data = torch.zeros(len(csv), self.num_features)
    self.labels = torch.zeros(len(csv), dtype=torch.long)
    for i in range(len(csv)):
      ptr = 0
      for j, feat in enumerate(self.metadata):
        if feat['type'] == 'continuous':
          self.data[i][ptr] = (
            (csv[j][i] - feat['min']) / (feat['max'] - feat['min']))
          ptr += 1
        else:
          idx = feat['values'].index(csv[j][i])
          assert idx >= 0
          self.data[i][ptr + idx] = 1
          ptr += len(feat['values'])
      self.labels[i] = '>50K' in csv[len(self.metadata)][i]

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, index):
    return self.data[index], self.labels[index]


def load_data(args, data, batch_size=256, test_batch_size=256):
  if data == 'MNIST':
    dummy_input = torch.randn(2, 1, 28, 28)
    mean, std = torch.tensor([0.0]), torch.tensor([1.0])
    train_data = datasets.MNIST(
        './data', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(
        './data', train=False, download=True, transform=transforms.ToTensor())
  elif data == 'CIFAR':
    cifar10_mean = [0.4914, 0.4822, 0.4465]
    # cifar10_std = [0.2023, 0.1994, 0.2010]
    cifar10_std = [0.2009, 0.2009, 0.2009]
    print('CIFAR std:', cifar10_std)
    mean = torch.tensor(cifar10_mean)
    std = torch.tensor(cifar10_std)
    dummy_input = torch.randn(2, 3, 32, 32)
    normalize = transforms.Normalize(mean = mean, std = std)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4, padding_mode='edge'),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    train_data = datasets.CIFAR10('./data', train=True, download=True,
      transform=transform)
    test_data = datasets.CIFAR10('./data', train=False, download=True,
      transform=transform_test)
  elif data == 'tinyimagenet':
    mean = torch.tensor([0.4802, 0.4481, 0.3975])
    std = torch.tensor([0.22, 0.22, 0.22])
    print('tinyimagenet std:', std)
    dummy_input = torch.randn(2, 3, 64, 64)
    normalize = transforms.Normalize(mean=mean, std=std)
    data_dir = 'data/tinyImageNet/tiny-imagenet-200'
    train_data = datasets.ImageFolder(
      data_dir + '/train',
      transform=transforms.Compose([
          transforms.RandomHorizontalFlip(),
          transforms.RandomCrop(64, 4, padding_mode='edge'),
          transforms.ToTensor(),
          normalize,
    ]))
    test_data = datasets.ImageFolder(
      data_dir + '/val',
      transform=transforms.Compose([
          transforms.ToTensor(),
          normalize
      ]))
  elif data == 'simpleCNN':
    dummy_input = torch.randn(2, 1, 8, 8)
    mean, std = torch.tensor([0.0]), torch.tensor([1.0])
    train_data = TensorDataset(
      0.5*torch.ones([100, 1, 8, 8]), torch.randint(0, 10, [100]))
    test_data = TensorDataset(
      0.5*torch.ones([100, 1, 8, 8]), torch.randint(0, 10, [100]))
  elif data == 'simpleData':
    dummy_input = torch.randn(2, 1, 4, 4)
    mean, std = torch.tensor([0.0]), torch.tensor([1.0])
    train_data = TensorDataset(
      0.5*torch.ones([100, 16]), torch.randint(0, 10, [100]))
    test_data = TensorDataset(
      0.5*torch.ones([100, 16]), torch.randint(0, 10, [100]))
  elif data == 'adult':
    train_data = Adult('./data', train=True)
    test_data = Adult('./data', train=False)
    dummy_input = torch.randn(2, train_data.num_features)
    mean, std = torch.tensor([0.]), torch.tensor([1.])
  else:
    raise ValueError(data)

  train_data = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True, pin_memory=True,
    num_workers=8)
  test_data = torch.utils.data.DataLoader(
    test_data, batch_size=test_batch_size, pin_memory=True, num_workers=8)

  train_data.mean = test_data.mean = mean
  train_data.std = test_data.std = std

  if data == 'adult':
    shape = (1, 1)
  else:
    shape = (1, -1, 1, 1)

  for loader in [train_data, test_data]:
    loader.mean, loader.std = mean, std
    if args.input_clipping:
      loader.data_max = torch.reshape((1. - mean) / std, shape)
      loader.data_min = torch.reshape((0. - mean) / std, shape)
    else:
      loader.data_max = torch.reshape((2. - mean) / std, shape)
      loader.data_min = torch.reshape((-1. - mean) / std, shape)

  dummy_input = dummy_input.to(args.device)

  return dummy_input, train_data, test_data
