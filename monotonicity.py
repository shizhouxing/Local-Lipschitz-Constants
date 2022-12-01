"""Monotonicity analysis."""
import torch
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import get_spec_matrix
from bab import bab_gradnorm
from tqdm import tqdm

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
