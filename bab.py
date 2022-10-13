import time
import torch
import numpy as np
from solver import LiRPAConvNet
from branching_domains import pick_out_batch, add_domain_parallel, Domain
from sortedcontainers import SortedList
from branching_heuristics import choose_branching

Visited = 0
debug = False

def batch_verification(d, net, batch, args=None):
  global Visited

  total_time = time.time()

  domains = pick_out_batch(
    d, batch=batch, device=net.x.device,
    intermediate=net.args.heuristic_intermediate)
  lAs, lAs_interm, uAs_interm, orig_lbs, orig_ubs, selected_domains = domains

  history = [sd.history for sd in selected_domains]

  branching_decision = choose_branching(
    orig_lbs, orig_ubs, net, lAs, heuristic=args.heuristic)

  lb_fully_splitted = float('inf')

  for i, dec in enumerate(branching_decision):
    if dec is None:
      # This domain has been fully splitted
      lb_fully_splitted = selected_domains[i].lower_bound
      orig_lbs = [lb[:i] for lb in orig_lbs]
      orig_ubs = [ub[:i] for ub in orig_ubs]
      branching_decision = branching_decision[:i]
      history = history[:i]
      selected_domains = selected_domains[:i]
      d.clear()
      print(f'Domain fully splitted: lb {lb_fully_splitted}, {i} remaining')
      if i == 0:
        return lb_fully_splitted, lb_fully_splitted
      else:
        break

  if debug:
    print(f'splitting decisions (first 10): {branching_decision[:10]}')

  ret = net.get_lower_bound(orig_lbs, orig_ubs, branching_decision)
  dom_ub, dom_lb, lAs, lAs_interm, uAs_interm, dom_lb_all, dom_ub_all = ret

  batch = len(branching_decision)
  # If intermediate layers are not refined or updated, we do not need to check
  # infeasibility when adding new domains.
  check_infeasibility = False

  unsat_list = add_domain_parallel(
    lA=lAs, lAs_intermediate=lAs_interm, uAs_intermediate=uAs_interm,
    lb=dom_lb, ub=dom_ub, lb_all=dom_lb_all, up_all=dom_ub_all,
    domains=d, selected_domains=selected_domains,
    branching_decision=branching_decision,
    check_infeasibility=check_infeasibility)

  # One unstable neuron split to two nodes
  Visited += (len(selected_domains) - len(unsat_list)) * 2
  total_time = time.time() - total_time
  assert len(d) > 0
  global_lb = d[0].lower_bound
  if debug:
    print(f'Current lb:{global_lb}')
    print(f'{Visited} neurons visited')
  return global_lb, lb_fully_splitted


def bab_gradnorm(model, x, grad_start, c, c_forward,
    opt_forward=True, args=None, timeout=None, bab=True):
  start = time.time()

  global debug
  debug = args.debug
  if timeout is None:
    timeout = args.timeout
  batch = args.batch_size
  global Visited
  Visited, global_ub = 0, np.inf
  forward_final_name = model.forward_final_name

  x_extra = (grad_start,)
  net = LiRPAConvNet(model, c=c, c_forward=c_forward,
    forward_final_name=forward_final_name, args=args)

  if not args.bab or not bab:
    # FIXME call `build_the_model` only once
    ret = net.build_the_model(x, x_extra=x_extra, opt_forward=opt_forward,
                              bab=False)
    if args.mono:
      lb = net.net[net.crown_start_nodes[1]].lower
      ub = net.net[net.crown_start_nodes[1]].upper
      return lb, ub
    else:
      return ret

  ret = net.build_the_model(x, x_extra=x_extra)
  global_ub, global_lb, lA, lA_interm, uA_interm, lbs, ubs, history = ret

  if isinstance(global_lb, torch.Tensor):
    global_lb = global_lb.item()

  candidate_domain = Domain(
    lA, lA_interm, uA_interm, global_lb, global_ub, lbs, ubs,
    history=history).to_cpu()
  domains = SortedList()
  domains.add(candidate_domain)

  ans = float('inf')

  while len(domains) > 0:
    with torch.no_grad():
      global_lb, lb_fully_splitted = batch_verification(
        domains, net, batch, args)
      ans = min(ans, lb_fully_splitted)

    if isinstance(global_lb, torch.Tensor):
      global_lb = global_lb.item()

    if len(domains) > args.max_domains:
      print('Reached maximum number of domains')
      break
    if time.time() - start > timeout:
      print('Time out!')
      break
    if debug:
      print(f'Cumulative time: {time.time() - start}\n')

  ans = min(global_lb, ans)

  del domains
  return ans
