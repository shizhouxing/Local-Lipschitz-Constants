import copy
import torch

class Domain:
  """
  Object representing a domain where the domain is specified by decision
  assigned to ReLUs.
  Comparison between instances is based on the values of
  the lower bound estimated for the instances.
  """

  def __init__(self, lA=None, lA_intermediate=None, uA_intermediate=None,
      lb=-float('inf'), ub=float('inf'),
      lb_all=None, up_all=None, depth=0, history=None):
    if history is None:
      history = []

    self.lA = lA
    self.lA_intermediate = lA_intermediate
    self.uA_intermediate = uA_intermediate
    self.lower_bound = lb
    self.upper_bound = ub
    self.lower_all = lb_all
    self.upper_all = up_all
    self.history = history
    self.valid = True
    self.depth = depth

  def __lt__(self, other):
    return self.lower_bound < other.lower_bound

  def __le__(self, other):
    return self.lower_bound <= other.lower_bound

  def __eq__(self, other):
    return self.lower_bound == other.lower_bound

  def to_cpu(self):
    # Transfer the content of this domain to cpu memory
    # (try to reduce memory consumption)

    def to_cpu_(obj):
      if isinstance(obj, list) or isinstance(obj, tuple):
        it = range(len(obj))
      elif isinstance(obj, dict):
        it = obj
      else:
        raise ValueError(obj)
      for i in it:
        if isinstance(obj[i], torch.Tensor):
          obj[i] = obj[i].to(device='cpu', non_blocking=True)
        elif type(obj[i]) in [list, tuple, dict]:
          to_cpu_(obj[i])

    to_cpu_([self.lA, self.lower_all, self.upper_all])

    return self


def add_domain_parallel(
    lA, lAs_intermediate, uAs_intermediate, lb, ub, lb_all, up_all, domains,
    selected_domains, branching_decision=None, decision_thresh=0,
    check_infeasibility=True):
  assert branching_decision is not None

  unsat_list = []
  batch = len(selected_domains)
  for i in range(batch):
    def add_domain(i, decision, selected_domain):
      infeasible = False
      if lb[i] < decision_thresh:
        if check_infeasibility:
          for (l, u) in zip(lb_all[i][1:-1], up_all[i][1:-1]):
            if (l-u).max() > 1e-6:
              infeasible = True
              print('Infeasible detected when adding to domain!')
              break

        if not infeasible:
          new_history = copy.deepcopy(selected_domain.history)
          new_history[decision[0]][0].append(decision[1])
          new_history[decision[0]][1].append(+1.0)

          # sanity check repeated split
          if decision[1] in selected_domain.history[decision[0]][0]:
            print('BUG!!! repeated split!')
            print(selected_domain.history)
            print(decision)
            raise RuntimeError

          child = Domain(
            lA[i], lAs_intermediate[i], uAs_intermediate[i], lb[i], ub[i],
            lb_all[i], up_all[i],
            depth=selected_domain.depth+1, history=new_history)
          domains.add(child)

    add_domain(i, branching_decision[i], selected_domains[i])
    add_domain(i + batch, branching_decision[i], selected_domains[i])

  return unsat_list

def pick_out_batch(
    domains, batch, intermediate=False, threshold=0, device='cuda'):
  """
  Pick the first batch of domains in the `domains` sequence
  that has a lower bound lower than `threshold`.

  Any domain appearing before the chosen one but having a lower_bound greater
  than the threshold is discarded.

  Returns: Non prunable CandidateDomain with the lowest reference_value.
  """
  assert batch > 0

  if torch.cuda.is_available():
    torch.cuda.synchronize()

  idx = 0
  batch = min(len(domains), batch)
  lAs, lAs_intermediate, uAs_intermediate = [], [], []
  lower_all, upper_all, selected_candidate_domains = [], [], []
  assert len(domains) > 0, 'Empty domains list.'

  while True:
    if len(domains) == 0:
      print(f'No domain left. Batch limit {batch} current batch: {idx}')
      break
    selected_candidate_domain = domains.pop(0)
    if (selected_candidate_domain.lower_bound < threshold
        and selected_candidate_domain.valid is True):
      idx += 1
      lAs.append(selected_candidate_domain.lA)
      lAs_intermediate.append(selected_candidate_domain.lA_intermediate)
      uAs_intermediate.append(selected_candidate_domain.uA_intermediate)
      lower_all.append(selected_candidate_domain.lower_all)
      upper_all.append(selected_candidate_domain.upper_all)
      selected_candidate_domains.append(selected_candidate_domain)
      selected_candidate_domain.valid = False  # set False to avoid another pop
      if idx == batch: break
    selected_candidate_domain.valid = False   # set False to avoid another pop

  batch = idx

  if batch == 0:
    return None, None, None, None, None

  lower_bounds = []
  for j in range(len(lower_all[0])):
    lower_bounds.append(torch.cat([lower_all[i][j]for i in range(batch)]))
  lower_bounds = [t.to(device=device, non_blocking=True) for t in lower_bounds]

  upper_bounds = []
  for j in range(len(upper_all[0])):
    upper_bounds.append(torch.cat([upper_all[i][j] for i in range(batch)]))
  upper_bounds = [t.to(device=device, non_blocking=True) for t in upper_bounds]

  # Reshape to batch first in each list.
  new_lAs, new_lAs_intermediate, new_uAs_intermediate = [], [], []
  for j in range(len(lAs[0])):
    if lAs[0][j] is not None:
      lA_ = torch.cat([lAs[i][j] for i in range(batch)])
    else:
      lA_ = None
    new_lAs.append(lA_.to(device=device, non_blocking=True) if lA_ is not None else None)
    if intermediate:
      lA_intermediate_ = torch.cat(
        [lAs_intermediate[i][j] for i in range(batch)]
      ) if lAs_intermediate[0][j] is not None else None
      uA_intermediate_ = torch.cat(
        [uAs_intermediate[i][j] for i in range(batch)]
      ) if uAs_intermediate[0][j] is not None else None
      new_lAs_intermediate.append(
        lA_intermediate_.to(device=device, non_blocking=True)
        if lA_intermediate_ is not None else None)
      new_uAs_intermediate.append(
        uA_intermediate_.to(device=device, non_blocking=True)
        if uA_intermediate_ is not None else None)
    else:
      new_lAs_intermediate.append([])
      new_uAs_intermediate.append([])

  # Non-contiguous bounds will cause issues,
  # so we make sure they are contiguous here.
  lower_bounds = [
    t if t.is_contiguous() else t.contiguous() for t in lower_bounds]
  upper_bounds = [
    t if t.is_contiguous() else t.contiguous() for t in upper_bounds]

  return (
    new_lAs, new_lAs_intermediate, new_uAs_intermediate,
    lower_bounds, upper_bounds, selected_candidate_domains)
