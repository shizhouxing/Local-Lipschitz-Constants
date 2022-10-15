""" Modules for gradients """
import torch
from torch.autograd import Function
from torch.nn import Module
import torch.nn.functional as F
from grad_bounds import BoundReluGrad

class SqrOp(Function):
  @staticmethod
  def symbolic(_, x):
    return _.op('grad::Sqr', x)

  @staticmethod
  def forward(ctx, x):
    return torch.square(x)

class Conv2dGrad(Module):
  def __init__(self, fw_module, weight, stride, padding, dilation, groups):
    super().__init__()
    self.weight = weight
    self.dilation = dilation
    self.groups = groups
    self.fw_module = fw_module

    assert isinstance(stride, list) and stride[0] == stride[1]
    assert isinstance(padding, list) and padding[0] == padding[1]
    assert isinstance(dilation, list) and dilation[0] == dilation[1]
    self.stride = stride[0]
    self.padding = padding[0]
    self.dilation = dilation[0]

  def forward(self, grad_last):
    output_padding0 = (
      int(self.fw_module.input_shape[2])
      - (int(self.fw_module.output_shape[2]) - 1) * self.stride
      + 2 * self.padding - 1 - (int(self.weight.size()[2] - 1) * self.dilation))
    output_padding1 = (
      int(self.fw_module.input_shape[3])
      - (int(self.fw_module.output_shape[3]) - 1) * self.stride
      + 2 * self.padding - 1 - (int(self.weight.size()[3] - 1) * self.dilation))

    return Conv2dGradOp.apply(
      grad_last, self.weight, self.stride, self.padding, self.dilation,
      self.groups, output_padding0, output_padding1)

class LinearGrad(Module):
  def __init__(self, weight):
    super().__init__()
    self.weight = weight

  def forward(self, grad_last):
    return F.linear(grad_last, self.weight.t())

class ReLUGradOp(Function):
  """ Local gradient of ReLU.

  Not including multiplication with gradients from other layers.
  """
  @staticmethod
  def symbolic(_, g, g_relu, g_relu_rev, preact):
    return _.op('grad::Relu', g, g_relu, g_relu_rev, preact)

  @staticmethod
  def forward(ctx, g, g_relu, g_relu_rev, preact):
    return g * BoundReluGrad.relu_grad(preact)

class ReLUGrad(Module):
  def forward(self, g, preact):
    g_relu = F.relu(g)
    g_relu_rev = -F.relu(-g)
    return ReLUGradOp.apply(g, g_relu, g_relu_rev, preact)

class ReshapeGrad(Module):
  def forward(self, grad_last, inp):
    return grad_last.reshape(
      grad_last.size(0), grad_last.size(1), *inp.shape[1:])

class FlattenGrad(Module):
  def __init__(self, in_shape):
    super().__init__()
    self.in_shape = in_shape

  def forward(self, grad_last):
    return torch.reshape(grad_last, [-1] + list(self.in_shape))

class Conv2dGradOp(Function):
  @staticmethod
  def symbolic(
      g, x, w, stride, padding, dilation, groups,
      output_padding0, output_padding1):
    return g.op(
      'grad::Conv2d', x, w, stride_i=stride, padding_i=padding,
      dilation_i=dilation, groups_i=groups, output_padding0_i=output_padding0,
      output_padding1_i=output_padding1)

  @staticmethod
  def forward(
      ctx, grad_last, w, stride, padding, dilation, groups, output_padding0,
      output_padding1):
    grad_shape = grad_last.shape
    grad = F.conv_transpose2d(
      grad_last.view(grad_shape[0] * grad_shape[1], *grad_shape[2:]), w, None,
      stride=stride, padding=padding, dilation=dilation,
      groups=groups, output_padding=(output_padding0, output_padding1))

    grad = grad.view((grad_shape[0], grad_shape[1], *grad.shape[1:]))
    return grad

class GradNorm(Module):
  def __init__(self, norm=1):
    super().__init__()
    self.norm = norm

  def forward(self, grad):
    grad = grad.view(grad.size(0), grad.size(1), -1)
    if self.norm == 1:
      # torch.norm is not supported in auto_LiRPA yet
      # use simpler operators for now
      return grad.abs().sum(dim=-1)
    elif self.norm == 2:
      return SqrOp.apply(grad).sum(dim=-1)
    else:
      raise NotImplementedError(self.norm)
