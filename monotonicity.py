"""Monotonicity analysis."""
import torch
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import get_spec_matrix
from bab import bab_gradnorm
from tqdm import tqdm

def monotonicity_mnist(args, model, test_data):
  """ Toy monotonicity analysis on MNIST """

  def verify(args, model, x, y):
    x = x.unsqueeze(0).cuda()
    label = torch.Tensor([y]).cuda().long()

    mono_lower = torch.zeros_like(x)
    mono_upper = torch.zeros_like(x)
    for i in range(len(x[0])):
      for j in range(len(x[0][0])):
        for k in range(len(x[0][0][0])):
          data_lb = x.clone()
          data_ub = x.clone()
          data_lb[0,i,j,k] = 0
          data_ub[0,i,j,k] = 1.0

          ptb = PerturbationLpNorm(
            norm=args.norm, eps=None, x_L=data_lb, x_U=data_ub)
          x = data = BoundedTensor(x, ptb)

          grad_start = torch.zeros(1, 1, args.num_classes).to(x)
          # TODO Only check the ground-truth label for now
          grad_start[0, 0, label.item()] = 1
          y = model(x, grad_start)
          model(x, grad_start)

          c = torch.ones(1, 1, 1).to(x)
          c_forward = get_spec_matrix(data, label, args.num_classes)

          model(x, grad_start)
          ret = bab_gradnorm(
            model, x, grad_start, c=-c, c_forward=c_forward, args=args)
          mono_lower[0,i,j,k] = ret[0][0,0, i*784 + j * 28 + k].detach().cpu()
          mono_upper[0,i,j,k] = ret[1][0,0, i*784 + j * 28 + k].detach().cpu()

    torch.save(mono_lower, f'mono_lower_{label.item()}_.torch')
    torch.save(mono_upper, f'mono_upper_{label.item()}_.torch')
    torch.save(x.squeeze(), f'input_{label.item()}_.torch')

  label_set = set()
  i=0
  while True:
    if test_data.dataset[i][1] in label_set:
      i += 1
      continue
    label_set.add(test_data.dataset[i][1])
    verify(args, model, *test_data.dataset[i])
    i += 1
    if len(label_set) == 10:
      break

def monotonicity(args, model, loader):
  """ New monotonicity analysis on the Adult dataset """
  num_features = loader.dataset[0][0].shape[0]
  monotonic_inc = torch.zeros(num_features)
  monotonic_dec = torch.zeros(num_features)

  # These are continuous features
  continuous_features = [0, 10, 27, 63, 64, 65]
  res = []
  for i in tqdm(range(args.num_examples)):
    data, label = loader.dataset[i]
    data, label = data.to(args.device), label.to(args.device)
    data = data.unsqueeze(0)
    label = torch.tensor([label])
    print(f'Example {i}: label {label}')
    assert data.ndim == 2
    res.append([])
    for j in continuous_features:
      data_lb = data.clone()
      data_ub = data.clone()
      data_lb[0, j] = 0
      data_ub[0, j] = 1
      ptb = PerturbationLpNorm(x_L=data_lb, x_U=data_ub)
      x = BoundedTensor(data, ptb)
      grad_start = torch.zeros(1, 1, args.num_classes).to(x)
      # Verify the score on label 1
      grad_start[0, 0, 1] = 1

      model(x, grad_start, final_node_name=model.forward_final_name)
      model(x, grad_start)
      c = torch.ones(1, 1, 1).to(x)
      c_forward = get_spec_matrix(data, label, args.num_classes)
      lb, ub = bab_gradnorm(model, x, grad_start, c=-c, c_forward=c_forward,
        args=args)
      if lb[0, 0, j] >= 0:
        monotonic_inc[j] += 1
        print('Increasing', j)
        res[-1].append((j, 'inc'))
      elif ub[0, 0, j] <= 0:
        monotonic_dec[j] += 1
        print('Decreasing', j)
        res[-1].append((j, 'dec'))

  print(res)
  print(monotonic_inc[continuous_features])
  print(monotonic_dec[continuous_features])

  torch.save(monotonic_inc, 'monotonic_inc.pt')
  torch.save(monotonic_dec, 'monotonic_dec.pt')
