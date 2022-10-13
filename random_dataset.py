"""Random datasets.

https://github.com/revbucket/lipMIP/blob/master/neural_nets/data_loaders.py
"""

import numpy as np
import random
import torch

class ParameterObject:
  def __init__(self, **kwargs):
    self.attr_list = []
    assert 'attr_list' not in kwargs
    for k, v in kwargs.items():
      setattr(self, k, v)
      self.attr_list.append(k)

  def change_attrs(self, **kwargs):
    new_kwargs = {}
    for attr in self.attr_list:
      if attr in kwargs:
        new_kwargs[attr] = kwargs[attr]
      else:
        new_kwargs[attr] = getattr(self, attr)
    return self.__class__(**new_kwargs)

class RandomKParameters(ParameterObject):
  """ Random K-cluster dataset
      Basic gist is to repeat the following process:
      - pick a bunch of random points
      - randomly select k of them to be 'leaders', randomly assign
        labels to these leaders
      - assign labels to the rest of the points by the label of their
        closest 'leader'
  PARAMETERS:
      - num points
      - num leaders (k)
  """

  flavor = 'randomk'

  def __init__(self, num_points, k, radius=None, dimension=2, num_classes=2):
    super(RandomKParameters, self).__init__(
      num_points=num_points, k=k, radius=radius, dimension=dimension,
      num_classes=num_classes)


class RandomDataset:
  """ Builds randomized d-dimensional, n-class datasets """
  def __init__(self, parameters, random_seed=None):
    assert isinstance(parameters, RandomKParameters)
    self.parameters = parameters
    self.dim = self.parameters.dimension
    if random_seed is None:
      random_seed = random.randint(1, 420 * 69)
    self.random_seed = random_seed
    np.random.seed(self.random_seed)
    self.generate_data()
    self.train_data = None
    self.val_data = None

  def generate_data(self):
    data, labels = self.generate_data_randomk()
    self.base_data = (data, labels)
    self.split_train_val(0.7)

  def split_train_val(self, train_prop):
    """ Generates two datasets, a training and validation dataset
    ARGS:
        train_prop: float in range [0, 1] - proportion of data used
                    ni the train set
    RETURNS:
        (train_set, test_set), where each is an iterable like
            [(examples, labels),...]
            where examples is one minibatch of the 2D Data
            and labels is one minibatch of the labels
    """

    perm = torch.randperm(self.parameters.num_points)
    num_train_data = int(train_prop * self.parameters.num_points)
    train_indices = perm[:num_train_data]
    test_indices = perm[num_train_data:]
    base_data, base_labels = self.base_data

    self.train_data = base_data[train_indices]
    self.train_labels = base_labels[train_indices]
    self.val_data = base_data[test_indices]
    self.val_labels = base_labels[test_indices]

  def generate_data_randomk(self):
    """ Generates random k-cluster data """
    num_points = self.parameters.num_points
    k = self.parameters.k
    num_classes = self.parameters.num_classes
    # first make data
    if getattr(self.parameters, 'radius') is not None:
      radius = self.parameters.radius
      data_points = self._generate_random_separated_data(num_points, radius,
                                                        self.dim)
      data_points = np.stack(data_points)
    else:
      data_points = np.random.uniform(size=(num_points, self.dim))

    # then pick leaders and assign them labels
    leader_indices = np.random.choice(num_points, size=(k), replace=False)
    random_labels = np.random.randint(low=0, high=num_classes, size=k,
                                      dtype=np.uint8)

    # and finally assign labels to everything else
    all_labels = np.zeros(num_points).astype(np.uint8)
    for i in range(num_points):
      min_leader_dist = np.inf
      min_leader_idx = None
      for j in range(k):
        leader = data_points[leader_indices[j]]
        dist = np.linalg.norm(leader - data_points[i])
        if dist < min_leader_dist:
          min_leader_dist = dist
          min_leader_idx = j
      all_labels[i] = random_labels[min_leader_idx]

    return torch.Tensor(data_points), torch.Tensor(all_labels).long()

  @classmethod
  def _generate_random_separated_data(cls, num_points, radius, dim):
    """ Generates num_points points in 2D at least radius apart
        from each other
    OUTPUT IS A LIST OF NUMPY ARRAYS, EACH OF SHAPE (dim,)
    """
    data_points = []
    while len(data_points) < num_points:
      point = np.random.uniform(size=(dim))
      if len(data_points) == 0:
        data_points.append(point)
        continue
      if min(np.linalg.norm(point - a) for a in data_points) > 2 * radius:
        data_points.append(point)
    return data_points
