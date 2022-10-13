import torch
import numpy as np
from auto_LiRPA.bound_ops import BoundLinear, BoundRelu, BoundReshape, BoundConv
from auto_LiRPA.bound_ops import BoundInput, BoundParams, BoundAdd, BoundSqr
from collections import deque
from grad_modules import LinearGrad, ReLUGrad, ReshapeGrad, Conv2dGrad
from grad_modules import GradNorm
from grad_bounds import BoundReluGrad, BoundConv2dGrad

custom_ops_grad = {
  'grad::Relu': BoundReluGrad,
  'grad::Conv2d': BoundConv2dGrad,
  'grad::Sqr': BoundSqr,
}

def convert_gradient_model(model, dummy_input, norm=np.inf):
  """Augment the computational graph with gradient computation."""

  final_node = model.final_node()
  if final_node.forward_value is None:
    model(dummy_input)
  output = final_node.forward_value
  if output.ndim != 2:
    raise NotImplementedError(
      'The model should have a 2-D output shape of (batch_size, output_dim)')
  output_dim = output.size(1)

  # Gradient values in `grad` may not be accurate. We do not consider gradient
  # accumulation from multiple succeeding nodes. We only want the shapes but not
  # the accurate values.
  grad = {}
  # Dummy values in grad_start
  grad_start = torch.ones(output.size(0), 1, output_dim, device=output.device)
  grad[final_node.name] = grad_start
  input_node_found = False

  # First BFS pass: traverse the graph, count degrees, and build gradient
  # layers.
  # Degrees of nodes.
  degree = {}
  # Original layer for gradient computation.
  layer_grad = {}
  # Input nodes in gradient computation in back propagation.
  input_nodes = {}
  # Dummy input values for gradient computation received.
  grad_input = {}
  # Extra nodes as arguments used for gradient computation.
  # They must match the order in grad_input.
  grad_extra_nodes = {}

  degree[final_node.name] = 0
  queue = deque([final_node])
  while len(queue) > 0:
    node = queue.popleft()

    grad_extra_nodes[node.name] = []
    input_nodes[node.name] = node.inputs
    if isinstance(node, BoundLinear):
      layer_grad[node.name] = LinearGrad(node.inputs[1].param)
      grad_input[node.name] = (grad[node.name],)
    elif isinstance(node, BoundRelu):
      layer_grad[node.name] = ReLUGrad()
      grad_input[node.name] = (grad[node.name], node.inputs[0].forward_value)
      # An extra node is needed to consider the state of ReLU activation
      grad_extra_nodes[node.name] = [node.inputs[0]]
    elif isinstance(node, BoundReshape):
      layer_grad[node.name] = ReshapeGrad()
      grad_input[node.name] = (grad[node.name], node.inputs[0].forward_value)
    elif isinstance(node, BoundInput):
      if input_node_found:
        raise NotImplementedError(
          'There must be exactly one BoundInput node, but found more than 1.')
      # TODO add an option to convert the graph without adding gradient norm
      dual_norm = 1. / (1. - 1. / norm) if norm != 1 else np.inf
      layer_grad[node.name] = GradNorm(norm=dual_norm)
      grad_input[node.name] = (grad[node.name],)
      input_node_found = True
    elif isinstance(node, BoundConv):
      layer_grad[node.name] = Conv2dGrad(
        node, node.inputs[1].param, node.stride, node.padding, node.dilation,
        node.groups)
      grad_input[node.name] = (grad[node.name],)
    else:
      raise NotImplementedError(f'Gradient for {node} is not supported')

    # Propagate gradients to the input nodes and update degrees.
    grad_next = layer_grad[node.name](*grad_input[node.name])
    if isinstance(grad_next, torch.Tensor):
      grad_next = [grad_next]
    if not isinstance(node, BoundInput):
      for i in range(len(grad_next)):
        grad[input_nodes[node.name][i].name] = grad_next[i]
        if not input_nodes[node.name][i].name in degree:
          degree[input_nodes[node.name][i].name] = 0
          queue.append(input_nodes[node.name][i])
        degree[input_nodes[node.name][i].name] += 1

  if not input_node_found:
    raise NotImplementedError(
      'There must be exactly one BoundInput node, but found none.')

  # Second BFS pass: build the backward computational graph
  grad_node = {}
  grad_node[final_node.name] = BoundInput(
    f'/grad{final_node.name}', grad_start)
  grad_node[final_node.name].name = f'/grad{final_node.name}'
  model.add_input_node(grad_node[final_node.name], index='auto')
  queue = deque([final_node])
  while len(queue) > 0:
    node = queue.popleft()
    nodes_op, nodes_in, nodes_out, _ = model._convert_nodes(
      layer_grad[node.name], grad_input[node.name])
    rename_dict = {}
    assert isinstance(nodes_in[0], BoundInput)
    rename_dict[nodes_in[0].name] = grad_node[node.name].name
    for i in range(1, len(nodes_in)):
      # Assume it's a parameter here
      new_name = f'/grad{node.name}/params{nodes_in[i].name}'
      rename_dict[nodes_in[i].name] = new_name
    for i in range(len(nodes_op)):
      # intermediate nodes
      if not nodes_op[i].name in rename_dict:
        new_name = f'/grad{node.name}/tmp{nodes_op[i].name}'
        rename_dict[nodes_op[i].name] = new_name
    if isinstance(node, BoundInput):
      assert len(nodes_out) == 1
      rename_dict[nodes_out[0].name] = '/grad_norm'
    else:
      for i in range(len(nodes_out)):
        assert not isinstance(node.inputs[i], BoundParams)
        rename_dict[nodes_out[i].name] = f'/grad{node.inputs[i].name}'

    model.rename_nodes(nodes_op, nodes_in, rename_dict)
    # Replace input nodes
    # grad_extra_nodes[node.name]: ReLU's input
    input_nodes_replace = (
      [model._modules[nodes_in[0].name]] + grad_extra_nodes[node.name])
    for i in range(len(input_nodes_replace)):
      for n in nodes_op:
        for j in range(len(n.inputs)):
          if n.inputs[j].name == nodes_in[i].name:
            n.inputs[j] = input_nodes_replace[i]
            # TODO remove redundant `input_name` in auto_LiRPA
            n.input_name[j] = input_nodes_replace[i].name
    model.add_nodes(nodes_op + nodes_in[len(input_nodes_replace):])

    if not isinstance(node, BoundInput):
      for i in range(len(nodes_out)):
        if input_nodes[node.name][i].name in grad_node:
          node_cur = grad_node[input_nodes[node.name][0].name]
          node_add = BoundAdd(
            attr=None, inputs=[ node_cur, nodes_out[i]], output_index=0,
            options={})
          node_add.name = f'{nodes_out[i].name}/add',
          grad_node[input_nodes[node.name][0].name] = node_add
        else:
          grad_node[input_nodes[node.name][0].name] = nodes_out[i]
        degree[input_nodes[node.name][i].name] -= 1
        if degree[input_nodes[node.name][i].name] == 0:
          queue.append(input_nodes[node.name][i])

  model(dummy_input, grad_start, final_node_name='/grad_norm')

  return model