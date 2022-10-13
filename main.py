import math
import time
import torch
import torch.nn as nn
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.utils import MultiAverageMeter, get_spec_matrix, logger
from auto_LiRPA.perturbations import PerturbationLpNorm
from datasets import load_data
from parser import parse_args
from bab import bab_gradnorm
from monotonicity import monotonicity
from lipbab.LipBaB import lipbab
from convert import conv_to_linear
from global_lip import global_lip
from utils import prepare_model
from gradient import convert_gradient_model, custom_ops_grad

def evaluate_clean(model, loader):
  print('Evaluating clean accuracy')
  model.eval()
  meter = MultiAverageMeter()
  with torch.no_grad():
    for (data, labels) in loader:
      data, labels = data.cuda(), labels.cuda()
      y = model(data)
      pred = y.argmax(dim=-1)
      meter.update('acc', (pred == labels).float().mean(), data.size(0))
  acc = meter.avg('acc')
  print(f'Clean accuracy {acc:.4f}')

def lirpa_local_lipschitz(model, data, labels, data_lb, data_ub, args=None):
  time_begin = time.time()

  ptb = PerturbationLpNorm(norm=args.norm, x_L=data_lb, x_U=data_ub)
  x = data = BoundedTensor(data, ptb)
  assert x.size(0) == 1 # batch size must be 1
  c = torch.ones(1, 1, 1).to(x) # For backward graph
  c_forward = get_spec_matrix(
    data, labels, args.num_classes) # For forward graph

  if args.single_class:
    # Only check the ground-truth label
    grad_start = torch.zeros(1, 1, args.num_classes).to(x)
    grad_start[0, 0, labels.item()] = 1
    model(x, grad_start, final_node_name=model.forward_final_name)
    model(x, grad_start)
    ret = bab_gradnorm(model, x, grad_start, c=-c, c_forward=c_forward,
      args=args)
    print(ret)
    return ret
  else:
    # Check all classes
    ans = 0
    bounds = []
    for j in range(args.num_classes):
      grad_start = torch.zeros(1, 1, args.num_classes).to(x)
      grad_start[0, 0, j] = 1
      model(x, grad_start, final_node_name=model.forward_final_name)
      model(x, grad_start)
      ret = -bab_gradnorm(model, x, grad_start, c=-c, c_forward=c_forward,
        opt_forward=(len(bounds)==0), args=args, bab=False)
      if args.norm == 2:
        ret = math.sqrt(ret)
      bounds.append(ret)
    bounds = torch.tensor(bounds)
    sort_label = torch.argsort(-bounds)
    time_remaining = args.timeout - (time.time() - time_begin)
    time_per_class = time_remaining / len(sort_label)
    for j in sort_label:
      time_begin_class = time.time()
      grad_start = torch.zeros(1, 1, args.num_classes).to(x)
      grad_start[0, 0, j] = 1
      model(x, grad_start, final_node_name=model.forward_final_name)
      model(x, grad_start)
      time_remaining = args.timeout - (time.time() - time_begin)
      timeout = min(
        time_per_class - (time.time() - time_begin_class), time_remaining)
      ret = -bab_gradnorm(
        model, x, grad_start,
        c=-c, c_forward=c_forward, args=args, timeout=timeout)
      if args.norm == 2:
        ret = math.sqrt(ret)
      print(f'class {j}, ret {ret}\n')
      bounds[j] = ret
      if time.time() - time_begin >= args.timeout:
        break
    print(f'Worst class {sort_label[0]}->{torch.argmax(bounds)},',
          f'label {labels.item()}')
    print(bounds)
    ans = bounds.max()
    return ans

def local_lipschitz(args, model, loader, eps=None):
  data_max, data_min, std = loader.data_max, loader.data_min, loader.std
  if args.device == 'cuda':
    data_min, data_max, std = data_min.cuda(), data_max.cuda(), std.cuda()
  model.eval()
  avg = 0
  begin = time.time()
  eps = (eps / std).view(1, -1, *([1]*(data_min.ndim - 2)))
  indices = range(args.start, args.start + args.num_examples)

  for i, idx in enumerate(indices):
    data, labels = loader.dataset[idx]
    data = data.unsqueeze(0)
    labels = torch.tensor([labels])
    print(f'Example {i}: labels {labels}')

    data, labels = data.to(args.device), labels.to(args.device)
    data_lb = torch.max(data - eps, data_min)
    data_ub = torch.min(data + eps, data_max)

    instance_begin = time.time()
    if args.method == 'global':
      ans = global_lip(model, args=args)
    elif args.method == 'lipbab':
      ans = lipbab(model, data, labels, data_lb, data_ub, args=args)
    else:
      ans = lirpa_local_lipschitz(
        model, data, labels, data_lb, data_ub, args=args)
    if ans is not None:
      print(ans)
      avg += ans
    print('time', time.time() - instance_begin)
    print('\n\n')
  avg_lip = avg / len(indices)
  avg_time = (time.time() - begin) / len(indices)
  print(f'avg_lip {avg_lip:.2f} avg_time {avg_time:.2f}')

if __name__ == '__main__':
  args = parse_args()

  logger.info('Arguments: %s', args)

  model_ori  = prepare_model(args)
  logger.info(f'Model structure: \n{str(model_ori)}'.format())
  # Batch size must be 1 to verify gradients
  dummy_input, train_data, test_data = load_data(
    args, args.data, batch_size=1, test_batch_size=1)

  if args.cnn_to_mlp:
    model_ori = conv_to_linear(model_ori, dummy_input.shape)
    dummy_input = dummy_input.view(dummy_input.size(0), -1)
    logger.info('CNN converted to MLP: \n%s', model_ori)
  model_ori.to(args.device)
  dummy_input = dummy_input.to(args.device)
  dummy_output = model_ori(dummy_input)
  logger.info('Converting the original model')
  conv_mode = 'patches'
  for layer in model_ori._modules.values():
    if isinstance(layer, nn.Conv2d) and layer.stride[0] != 1:
      logger.info(
        'Using matrix mode due to convolutional layers with stride != 1')
      conv_mode = 'matrix'

  bound_opts = {
      'optimize_bound_args': {
          'ob_iteration': args.ob_iteration,
          'ob_lr_decay': args.ob_lr_decay,
          'ob_lr': args.ob_lr,
          'ob_no_float64_last_iter': True,
      },
      'sparse_intermediate_bounds': True,
      'sparse_conv_intermediate_bounds': True,
      'sparse_intermediate_bounds_with_ibp': True,
      'sparse_features_alpha': False,
      'sparse_spec_alpha': False,
      'conv_mode': conv_mode,
      'lip_method': args.method,
  }
  if args.method == 'recurjac':
    bound_opts['recurjac'] = True
  model = BoundedModule(model_ori, dummy_input, bound_opts=bound_opts,
    device=args.device, custom_ops=custom_ops_grad)

  if args.norm != np.inf:
    logger.warning('Using norm other than inf is not recommended.')
  convert_gradient_model(model, dummy_input, norm=args.norm)
  model.forward_final_name = model.final_name
  model.final_name = '/grad_norm'
  model.output_name = ['/grad_norm']
  print('\n\n\n')

  logger.info('Starting computing...')
  if args.mono:
    monotonicity(args, model, test_data)
  elif args.clean:
    evaluate_clean(model_ori, test_data)
  else:
    local_lipschitz(args, model, test_data, eps=args.eps)
