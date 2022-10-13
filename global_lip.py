"""Global Lipschitz constant."""

def global_lip(model, args=None):
  assert args.norm == float('inf')

  ans = 1.0

  for p in model.named_parameters():
    if 'weight' in p[0]:
      weight = p[1]
      if weight.ndim == 2:
        ans *= (weight.abs().sum(dim=-1)).max()
      else:
        ans *= weight.view(weight.size(0), -1).abs().sum(dim=-1).max()

  return ans
