import torch
import numpy as np
from torch.nn import functional as F
from auto_LiRPA.bound_ops import BoundRelu

@torch.no_grad()
def choose_branching(
    lower_bounds, upper_bounds, net, lAs, heuristic='area',
    filtering=False, k=3):
  topk = k if filtering else 1
  batch = len(lower_bounds[0])

  score = [None] * len(net.split_nodes)

  name_id = {}
  for i, node in enumerate(net.split_nodes):
    name_id[node.name] = i

  if heuristic == 'babsr':
    # Compute scores on the backward graph first
    for i, node in enumerate(net.split_nodes):
      if node.subgraph == 'backward':
        A = lAs[i].clamp(max=0)
        lb = lower_bounds[i].clamp(max=0)
        ub = upper_bounds[i].clamp(min=0)
        if isinstance(node, BoundRelu):
          intercept = -(
            lb * (F.relu(ub) - F.relu(lb)) / (ub - lb).clamp(min=1e-12))
          score[i] = (-A * intercept).view(batch, -1)
        else:
          score[i] = torch.zeros_like(lower_bounds[i]).view(batch, -1)

  # Compute scores for the forward graph
  for i, node in enumerate(net.split_nodes):
    if node.subgraph == 'forward':
      if node.backward_relu_1 is None:
        score[i] = torch.zeros_like(lower_bounds[i]).view(batch, -1)
      else:
        # baseline
        if heuristic == 'babsr':
          score[i] = score[node.backward_relu_1] + score[node.backward_relu_2]
        elif heuristic == 'area':
          lb_backward = lower_bounds[node.backward_relu_1]
          ub_backward = upper_bounds[node.backward_relu_1]
          # constant 0.5 is omitted
          score[i] = (
            -lb_backward * lb_backward.abs() +
            ub_backward * ub_backward.abs()).view(batch, -1)
          score[i] *= (
            lAs[node.backward_relu_1].abs() +
            lAs[node.backward_relu_2].abs()).view(batch, -1)
          mask_unstable = torch.logical_and(
            lower_bounds[i] <= 0, upper_bounds[i] >= 0).view(batch, -1)
          # Only split unstable nodes
          score[i] = ((score[i] + 1) * mask_unstable)
        else:
          raise NotImplementedError(heuristic)

  final_decision = []
  # Use score_length to convert an index to its layer and offset.
  score_length = np.cumsum(
    [len(score[i][0]) for i in range(net.num_branched_nodes)])
  score_length = np.insert(score_length, 0, 0)
  # Flatten the scores vector.
  all_score = torch.cat(score[:net.num_branched_nodes], dim=1)
  # Select top-k candidates among all layers for two kinds of scores.
  score_idx = torch.topk(all_score, topk)
  # These indices are the indices for the top-k scores in flatten
  score_idx_indices = score_idx.indices.cpu()

  if not filtering:
    decision_index = score_idx_indices[:, 0]
    for i, l in enumerate(decision_index):
      # Go over each element in this batch.
      l = l.item()
      if all_score[i, l] == 0:
        final_decision.append(None)
      else:
        # Recover the (layer, idx) from the flattend array.
        layer = np.searchsorted(score_length, l, side='right') - 1
        idx = l - score_length[layer]
        final_decision.append([layer, idx])
  else:
    # kFSB-like filtering for the branching heuristic
    k_decision = []
    k_ret = torch.empty(size=(topk, batch), device=lower_bounds[0].device)
    for k in range(topk):
      # top-k candidates from the slope scores.
      decision_index = score_idx_indices[:, k]
      # Find which layer and neuron this topk gradient belongs to.
      decision_max_ = []
      for l in decision_index:
        # Go over each element in this batch.
        l = l.item()
        # Recover the (layer, idx) from the flattend array.
        layer = np.searchsorted(score_length, l, side='right') - 1
        idx = l - score_length[layer]
        decision_max_.append([layer, idx])
      k_decision.append(decision_max_)
      k_ret_lbs = net.update_bounds_parallel(
        lower_bounds, upper_bounds, k_decision[-1], shortcut=True)
      # No need to set slope next time; we do not optimize the slopes.
      # build mask indicates invalid scores (stable neurons), batch wise,
      # 1: invalid
      mask_score = (score_idx.values[:, k] <= 1e-4).float()
      k_ret[k] = torch.min(
        (k_ret_lbs.view(-1) - mask_score.repeat(2) * 1e30).reshape(2, -1),
        dim=0).values

    # k_ret has shape (top-k, batch) and we take the score eveluated using bound
    # propagation based on the top-k choice.
    i_idx = k_ret.max(0)
    rets = i_idx.values.cpu().numpy()
    rets_indices = i_idx.indices.cpu().numpy()
    # Given the indices of the max score, find the corresponding decision.
    decision_tmp = [k_decision[rets_indices[ii]][ii] for ii in range(batch)]

    # regular kfsb, select the top 1 decision from k
    for b in range(batch):
      # make sure this potential split is valid
      if rets[b] > -1e20:
        final_decision.append(decision_tmp[b])
      else:
        final_decision.append(None)

  return final_decision
