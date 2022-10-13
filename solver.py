import time
import torch
from auto_LiRPA import BoundedTensor
from auto_LiRPA.bound_ops import BoundRelu, BoundNeg, BoundAbs, BoundSqr
from auto_LiRPA.perturbations import PerturbationLpNorm

class LiRPAConvNet:
  def __init__(self, model, c, c_forward, forward_final_name, args):
    self.net = model
    self.net.eval()
    self.c = c
    self.c_forward = c_forward
    self.forward_final_name = forward_final_name
    self.args = args

  def get_lower_bound(self, pre_lbs, pre_ubs, split):
    start = time.time()
    ret = self.update_bounds_parallel(pre_lbs, pre_ubs, split)
    lower_bounds, upper_bounds, lAs, lAs_intermediate, uAs_intermediate = ret
    end = time.time()
    if self.args.debug:
      print('batch bounding time: ', end - start)
    return (
      [i[-1].item() for i in upper_bounds],
      [i[-1].item() for i in lower_bounds],
      lAs, lAs_intermediate, uAs_intermediate, lower_bounds, upper_bounds)

  def get_candidate(self, lb, ub):
    lower_bounds, upper_bounds = [], []
    for node in self.split_nodes:
      node_pre = node.inputs[0]
      if not node_pre.perturbed:
        lower_bounds.append(node_pre.forward_value.detach())
        upper_bounds.append(node_pre.forward_value.detach())
      else:
        lower_bounds.append(node_pre.lower.detach())
        upper_bounds.append(node_pre.upper.detach())
    # Also add the bounds on the final thing
    lower_bounds.append(lb.view(1, -1).detach())
    upper_bounds.append(ub.view(1, -1).detach())
    return lower_bounds, upper_bounds

  def get_lAs_intermediate(self, A_dict, batch_size):
    A_dict = A_dict[self.crown_start_nodes[1]]
    lAs_intermediate = []
    uAs_intermediate = []
    if self.args.heuristic_intermediate:
      assert not self.args.fix_preact_gradnorm
      for i in range(batch_size):
        lAs_, uAs_ = [], []
        for node in self.split_nodes:
          lA_this = A_dict[node.name]['lA'] if node.name in A_dict else None
          lAs_.append(
            lA_this[:, i].unsqueeze(0).cpu() if lA_this is not None else None)
          uA_this = A_dict[node.name]['uA'] if node.name in A_dict else None
          uAs_.append(
            uA_this[:, i].unsqueeze(0).cpu() if uA_this is not None else None)
        lAs_intermediate.append(lAs_)
        uAs_intermediate.append(uAs_)
    else:
      lAs_intermediate = uAs_intermediate = [[] for i in range(batch_size)]
    return lAs_intermediate, uAs_intermediate

  def _fix_missing_lA(self):
    for node in self.net._modules.values():
      if (isinstance(node, BoundNeg) and node.lA is not None and
          isinstance(node.inputs[0], BoundRelu) and node.inputs[0].lA is None):
        # This propagation may have been skipped by auto_LiRPA
        # if the node is not perturbed
        node.inputs[0].lA = -node.lA

  def get_lAs(self, A_dict, batch_size=1):
    self._fix_missing_lA()
    lAs = [[
      (node.lA[0] if node.lA is not None else None)
      for node in self.split_nodes]]
    As = self.get_lAs_intermediate(A_dict, batch_size=batch_size)
    lAs_intermediate, uAs_intermediate = As
    return lAs, lAs_intermediate, uAs_intermediate

  def update_bounds_parallel(
      self, pre_lb_all=None, pre_ub_all=None, decision=None, shortcut=False):
    """Main function for computing bounds after BaB in Beta-CROWN."""
    batch = len(decision)
    batch_size = batch * 2

    # initial results with empty list
    ret_l = [[] for _ in range(batch_size)]
    ret_u = [[] for _ in range(batch_size)]

    # pre_ub_all[:-1] means pre-set bounds for all intermediate layers
    with torch.no_grad():
      # Setting the neuron upper/lower bounds with a split to 0.
      indices_batch = [[] for _ in range(len(pre_lb_all) - 1)]
      indices_neuron = [[] for _ in range(len(pre_lb_all) - 1)]
      for i in range(batch):
        d, idx = decision[i][0], decision[i][1]
        # We save the batch, and neuron number for each split, and will
        # set all corresponding elements in batch.
        indices_batch[d].append(i)
        indices_neuron[d].append(idx)
      indices_batch = [
          torch.as_tensor(t).to(device=self.net.device, non_blocking=True)
          for t in indices_batch]
      indices_neuron = [
          torch.as_tensor(t).to(device=self.net.device, non_blocking=True)
          for t in indices_neuron]
      upper_bounds = [
        ub.repeat(2, *([1]*(ub.ndim-1))) for ub in pre_ub_all[:-1]]
      lower_bounds = [
        lb.repeat(2, *([1]*(lb.ndim-1))) for lb in pre_lb_all[:-1]]
      # Only the last element is used later.
      pre_lb_last = pre_lb_all[-1].repeat(2, 1)
      pre_ub_last = pre_ub_all[-1].repeat(2, 1)
      new_candidate = {}
      for d in range(len(lower_bounds)):
        # for each layer except the last output layer
        if len(indices_batch[d]):
          # We set lower = 0 in first half batch, and upper = 0 in second half.
          # Due to Clarke gradient for ReLU, use 1e-9 and -1e-9 rather than 0.
          if self.split_nodes[d].subgraph == 'forward':
            threshold = 1e-9
          else:
            threshold = 0

          lower_bounds[d].view(batch_size, -1)[
            indices_batch[d], indices_neuron[d]] = threshold
          upper_bounds[d].view(batch_size, -1)[
            indices_batch[d], indices_neuron[d]].clamp_(min=threshold)
          lower_bounds[d].view(batch_size, -1)[
            indices_batch[d] + batch, indices_neuron[d]].clamp_(max=threshold)
          upper_bounds[d].view(batch_size, -1)[
            indices_batch[d] + batch, indices_neuron[d]] = -threshold

        # For abs, force to recompute intermediate bounds
        # TODO splitting on abs
        name = self.split_nodes[d].inputs[0].name
        # if not name in self.crown_start_nodes:
        #   new_candidate[name] = [lower_bounds[d], upper_bounds[d]]
        new_candidate[name] = [lower_bounds[d], upper_bounds[d]]

    # create new_x here since batch may change
    assert isinstance(self.x.ptb, PerturbationLpNorm)
    ptb = PerturbationLpNorm(
      norm=self.x.ptb.norm, eps=self.x.ptb.eps,
      x_L=self.x.ptb.x_L.repeat(batch_size, 1, 1, 1),
      x_U=self.x.ptb.x_U.repeat(batch_size, 1, 1, 1))
    new_x = BoundedTensor(self.x.data.repeat(batch_size, 1, 1, 1), ptb)

    new_x_extra = tuple(
      [x.data.repeat(batch_size, 1, 1, 1) for x in self.x_extra])
    c = self.c.repeat(new_x.shape[0], 1, 1)

    # FIXME reuse
    if shortcut:
      with torch.no_grad():
        lb, _ = self.net.compute_bounds(
          x=(new_x,)+new_x_extra, C=c, method='CROWN',
          intermediate_layer_bounds=new_candidate, bound_upper=False)
      return lb

    new_candidate = {}
    for d in range(len(lower_bounds)):
      name = self.split_nodes[d].inputs[0].name
      if not name in self.crown_start_nodes or self.args.fix_preact_gradnorm:
        new_candidate[name] = [lower_bounds[d], upper_bounds[d]]
    # We want pre-activation bounds for BoundAbs
    ret = self.net.compute_bounds(
      x=(new_x,)+new_x_extra, C=c, method='CROWN',
      intermediate_layer_bounds=new_candidate, bound_upper=False,
      return_A=self.args.heuristic_intermediate,
      needed_A_dict=self.needed_A_dict
    )
    if self.args.heuristic_intermediate:
      lb, _, A_dict = ret
    else:
      lb, _ = ret

    if self.args.heuristic_intermediate:
      lAs_interm, uAs_interm = self.get_lAs_intermediate(A_dict, batch_size)
    else:
      lAs_interm = uAs_interm = [[] for i in range(batch_size)]

    if not self.args.fix_preact_gradnorm:
      # Update pre-activation bounds for BoundAbs/BoundSqr
      for d in range(len(lower_bounds)):
        name = self.split_nodes[d].inputs[0].name
        if name in self.crown_start_nodes:
          # Branched bounds
          old_lb, old_ub = lower_bounds[d], upper_bounds[d]
          # Intermediate bounds from the latest CROWN
          new_lb = self.net[name].lower.detach()
          new_ub = self.net[name].upper.detach()
          # Take the tigher one
          new_candidate[name] = [
            torch.max(old_lb, new_lb), torch.min(old_ub, new_ub)]
      # Recompute bounds with update pre-activation bounds on BoundAbs/BoundSqr
      lb, _ = self.net.compute_bounds(
        x=(new_x,)+new_x_extra, C=c, method='CROWN',
        intermediate_layer_bounds=new_candidate, bound_upper=False)
    ub = lb + 1e9

    # Collect results, on CPU
    with torch.no_grad():
      lb, ub = lb.to(device='cpu'), ub.to(device='cpu')
      lAs = []
      self._fix_missing_lA()
      for i in range(batch_size):
        lAs.append([
          (node.lA[0, i:i+1].detach().cpu() if node.lA is not None else None)
          for node in self.split_nodes])
      lower_bounds_new, upper_bounds_new = [], []
      for node in self.split_nodes:
        lower_bounds_new.append(node.inputs[0].lower.detach().cpu())
        upper_bounds_new.append(node.inputs[0].upper.detach().cpu())
      lower_bounds_new.append(lb.detach().cpu().view(batch * 2, -1).detach())
      upper_bounds_new.append(ub.detach().cpu().view(batch * 2, -1).detach())
      lower_bounds_new[-1] = torch.max(
        lower_bounds_new[-1], pre_lb_last.detach().cpu())
      upper_bounds_new[-1] = torch.min(
        upper_bounds_new[-1], pre_ub_last.detach().cpu())
      for i in range(batch):
        ret_l[i] = [j[i:i + 1] for j in lower_bounds_new]
        ret_l[i + batch] = [
          j[i + batch:i + batch + 1] for j in lower_bounds_new]
        ret_u[i] = [j[i:i + 1] for j in upper_bounds_new]
        ret_u[i + batch] = [
          j[i + batch:i + batch + 1] for j in upper_bounds_new]

    return ret_l, ret_u, lAs, lAs_interm, uAs_interm

  def build_the_model(self, x, x_extra=None, opt_forward=True, bab=True):
    self.x = x
    self.x_extra = x_extra

    if self.args.method == 'ibp':
      lb, ub = self.net.compute_bounds(
        method='IBP', C=self.c, x=(x,) + x_extra)
      return lb[-1].item()

    intermediate_bounds = {}
    # Optimize bounds using the forward graph
    if not self.args.no_optimize and opt_forward:
      self.net.compute_bounds(
        method='CROWN-Optimized',
        C=self.c_forward, x=(x,) + x_extra, bound_upper=False,
        final_node_name=self.forward_final_name)
    for node in self.net._modules.values():
      if hasattr(node, 'lower') and node.lower is not None:
        intermediate_bounds[node.name] = (node.lower, node.upper)

    # Recompute intermediate bounds for these nodes during bab
    self.crown_start_nodes = [self.net.final_name]
    # For gradient norm
    for node in self.net._modules.values():
      if type(node) in [BoundAbs, BoundSqr]:
        self.crown_start_nodes.append(node.inputs[0].name)
    assert len(self.crown_start_nodes) == 2
    assert self.crown_start_nodes[0] == '/grad_norm'

    if not bab:
      lb, ub = self.net.compute_bounds(
        method='CROWN', C=self.c, x=(x,) + x_extra, bound_upper=False,
        intermediate_layer_bounds=intermediate_bounds
      )
      print('Initial crown:', lb)
      return lb[-1].item()

    # No need to optimize bounds on the backward graph
    self.needed_A_dict = {}
    for node in self.crown_start_nodes:
      self.needed_A_dict[node] = [ ]
    lb, ub, A_dict = self.net.compute_bounds(
      method='CROWN', C=self.c, x=(x,) + x_extra, bound_upper=False,
      intermediate_layer_bounds=intermediate_bounds,
      return_A=True, needed_A_dict=self.needed_A_dict
    )
    print('Initial crown:', lb)

    # A ReLU node is added if its pre-activation bounds will be used
    self.relus = [node for node in self.net.relus if node.inputs[0].used]
    self.split_nodes = []
    self.gn_idx = None
    for node in self.relus:
      if 'grad' in node.name:
        node.subgraph = 'backward'
      else:
        node.subgraph = 'forward'
        node.branch = True
        self.split_nodes.append(node)
    # Nodes for gradient norm that may also be splitted
    for node in self.net._modules.values():
      if type(node) in [BoundAbs, BoundSqr]:
        node.branch = self.args.branch_gn
        assert self.gn_idx is None
        # index of abs/sqr
        self.gn_idx = len(self.split_nodes)
        self.split_nodes.append(node)
        node.subgraph = 'backward'
    # Also add ReLU in the backward graph, which will help score calculation
    for node in self.relus:
      if node.subgraph == 'backward':
        node.branch = False
        self.split_nodes.append(node)

    self.needed_A_dict = {}
    for node in self.crown_start_nodes:
      self.needed_A_dict[node] = [ node.name for node in self.split_nodes ]

    idx_split_nodes = {}
    for i, node in enumerate(self.split_nodes):
      idx_split_nodes[node.name] = i
    for i, node in enumerate(self.split_nodes):
      if node.subgraph == 'forward':
        preact = node.inputs[0]
        grad_node = self.net[preact.output_name[1]]
        assert isinstance(grad_node.inputs[1], BoundRelu) # unstable A_neg
        assert isinstance(grad_node.inputs[2], BoundNeg) # unstable A_pos
        # two corresponding ReLU neurons in the backward graph
        if grad_node.inputs[1].name in idx_split_nodes:
          node.backward_relu_1 = idx_split_nodes[grad_node.inputs[1].name]
          node.backward_relu_2 = idx_split_nodes[
            grad_node.inputs[2].inputs[0].name]
        else:
          node.backward_relu_1 = node.backward_relu_2 = None

    lb, ub = self.get_candidate(lb, lb + 1e9)  # primals are better upper bounds
    As = self.get_lAs(A_dict, batch_size=len(lb[0]))
    lA, lA_intermediate, uAs_intermediate = As
    history = [[[], []] for _ in range(len(self.split_nodes))]

    # Assuming all the branched nodes are among the first nodes in split_nodes
    self.num_branched_nodes = len(
      [node for node in self.split_nodes if node.branch])

    return (
      ub[-1].item(), lb[-1].item(), lA[0],
      lA_intermediate[0], uAs_intermediate[0], lb, ub, history)
