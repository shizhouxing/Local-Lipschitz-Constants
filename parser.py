"""Arguments."""
import argparse

def add_arguments_lipschitz(parser):
  parser.add_argument('--method', type=str, default='ours', choices=[
                      'ours', 'lipbab', 'ibp', 'global', 'recurjac'])
  parser.add_argument('--use-recurjac-model', action='store_true')
  parser.add_argument('--norm', type=float, default=float('inf'))
  parser.add_argument('--bab', action='store_true')
  parser.add_argument('--eps', type=float)
  parser.add_argument('--heuristic', type=str, default='area',
                      choices=['babsr', 'area'])
  parser.add_argument('--filtering', action='store_true')
  parser.add_argument('--branching-candidates', type=int, default=3)
  parser.add_argument('--start', type=int, default=0)
  parser.add_argument('--num-examples', '--num_example', type=int, default=1)
  parser.add_argument('--batch-size', type=int, default=128,
                      help='Batch size during bab.')
  parser.add_argument('--timeout', type=int, default=60)
  parser.add_argument('--max-domains', type=int, default=200000)
  parser.add_argument('--ob-iteration', '--opt-steps', type=int, default=20)
  parser.add_argument('--ob-lr-decay', type=float, default=0.98)
  parser.add_argument('--ob-lr', type=float, default=0.5)
  parser.add_argument('--no-optimize', action='store_true')
  parser.add_argument('--cnn-to-mlp', action='store_true',
                      help='Convert CNN to MLP for baselines.')
  parser.add_argument('--mono', action='store_true', help='Monotonicity.')
  parser.add_argument('--clean', action='store_true', help='Clean evaluation.')
  parser.add_argument('--fix-preact-gradnorm', action='store_true',
                      help='During bab, fix pre-activation bounds for '
                           'gradient norm unless they are splitted;'
                           'otherwise, recompute pre-activation bounds.')
  parser.add_argument('--branch-gn', action='store_true',
                      help='branch on grad norm node (abs/sqr)')
  parser.add_argument('--heuristic-intermediate', action='store_true',
                      help='Consider intermediate bounds in heuristic')

def add_arguments_data(parser):
  parser.add_argument('--data', type=str, default='MNIST',
                      choices=['MNIST', 'CIFAR', 'tinyimagenet', 'simpleData',
                               'simpleCNN', 'adult'])
  parser.add_argument('--num-classes', type=int, default=10)
  parser.add_argument('--input-clipping', action='store_true',
                      help='Clip input range to [0,1]')

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--load', type=str, default='')
  parser.add_argument('--device', type=str, default='cuda',
                      choices=['cpu', 'cuda'], help='Use CPU or CUDA.')
  parser.add_argument('--model', type=str, default='cnn')
  parser.add_argument('--model-params', type=str, default='')

  add_arguments_data(parser)
  add_arguments_lipschitz(parser)
  args = parser.parse_args()

  return args
