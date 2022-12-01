import ast
import onnx
import onnx2pytorch
import torch
import torch.nn as nn
from models import * # pylint: disable=wildcard-import,unused-wildcard-import
from convert import convert_recurjac_model
from auto_LiRPA.utils import MultiAverageMeter

ce_loss = nn.CrossEntropyLoss()

def update_meter(meter, regular_ce, robust_loss, regular_err, robust_err, batch_size):
  meter.update('CE', regular_ce, batch_size)
  if robust_loss is not None:
    meter.update('Rob_Loss', robust_loss, batch_size)
  if regular_err is not None:
    meter.update('Err', regular_err, batch_size)
  if robust_err is not None:
    meter.update('Rob_Err', robust_err, batch_size)

def parse_opts(string):
  opts = string.split(',')
  params = {}
  for opt in opts:
    if opt.strip():
      key, val = opt.split('=')
      try:
        value = ast.literal_eval(val)
      except: # pylint: disable=bare-except
        value = val
      if type(value) not in [int, float, bool]:
        value = val
      params[key] = value
  return params

def prepare_model(args):
  model = args.model

  if args.data == 'MNIST':
    input_shape = (1, 28, 28)
  elif args.data == 'CIFAR':
    input_shape = (3, 32, 32)
  elif args.data == 'tinyimagenet':
    input_shape = (3, 64, 64)
  elif args.data == 'simpleData':
    input_shape = (1, 16, 16)
  elif args.data == 'simpleCNN':
    input_shape = (1, 16, 16)
  elif args.data == 'adult':
    input_shape = (108,)
  else:
    raise NotImplementedError(args.data)

  if args.load and args.load.endswith('.onnx'):
    print(f'Loading ONNX model {args.load}')
    onnx_model = onnx.load(args.load)
    pytorch_model = onnx2pytorch.ConvertModel(onnx_model)
    layers = []
    for layer in pytorch_model._modules.values():
      if 'Flatten' in str(type(layer)):
        layers.append(Flatten())
      else:
        layers.append(layer)
    model_ori = nn.Sequential(*layers)
    return model_ori

  if args.use_recurjac_model:
    model_ori = convert_recurjac_model(args.model)
  else:
    if len(input_shape) == 1:
      model_ori = globals()[model](
        in_dim=input_shape[0], **parse_opts(args.model_params))
    else:
      model_ori = globals()[model](
        in_ch=input_shape[0], in_dim=input_shape[1],
        **parse_opts(args.model_params))
  if args.load:
    checkpoint = torch.load(args.load)
    if 'state_dict' in checkpoint:
      state_dict = checkpoint['state_dict']
    else:
      state_dict = checkpoint
    model_ori.load_state_dict(state_dict, strict=False)

  return model_ori

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
