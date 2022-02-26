
import argparse

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(123456)

from ucb import UCB
from klucb import KLUCB
from moss import MOSS
from etc import ETC

parser = argparse.ArgumentParser(description='Bandit Algorithms')
parser.add_argument('--algo', default='ucb', help='Which algo to run? ETC, UCB, MOSS or KL-UCB')
parser.add_argument('--num_arms', nargs = '+', default='2 5 7 10', help='Number of arms in the experiment (int or vector)')
parser.add_argument('--horizon', nargs='+', default='20000 40000', help='Horizon or total number of steps in the experiment (int or vector)')

args = parser.parse_args()
num_arms_list = [int(i) for i in args.num_arms.split(' ')]
num_steps_list = [int(i) for i in args.horizon.split(' ')]
algo_name = args.algo.upper()

m_exp = [100, 500, 1000] # number of times each arm is pulled in pure exploration phase

## Plot the regret curves
fig, axes = plt.subplots(2, len(num_arms_list)//2, figsize=(10, 10))

for idx, k in enumerate(num_arms_list):
  ### Generating distribution for each arm, we can choose any we like.
  ### We use dirichlet distribution in this case
  alpha = np.random.randint(1, k+1, k)
  arm_means = np.random.dirichlet(alpha, size = 1).squeeze(0)
  legend = []
  if args.algo == 'etc':
      for n in num_steps_list:
        for m in m_exp:
            regret = ETC(arm_means, k, n, m)
            cum_regret = np.cumsum(regret, axis = 0)
            avg_regret = np.mean(cum_regret, axis = 1)
            axes[idx//2, idx%2].plot(np.arange(n), avg_regret)
            legend.append(f'n={n}, m={m}')
  else:
    for n in num_steps_list:
        if args.algo == 'ucb':
            regret = UCB(arm_means, k, n)
        elif args.algo == 'moss':
            regret = MOSS(arm_means, k, n)
        else:
            regret = KLUCB(arm_means, k, n)
        cum_regret = np.cumsum(regret, axis = 0)
        avg_regret = np.mean(cum_regret, axis = 1)
        axes[idx//2, idx%2].plot(np.arange(n), avg_regret)
        legend.append(f'n={n}')
  axes[idx//2, idx%2].set_title(f'Number of arms={k}')
  axes[idx//2, idx%2].legend(legend)
  axes[idx//2, idx%2].set_xlabel('Number of time steps', fontsize = 8)
  axes[idx//2, idx%2].set_ylabel('Cumulative Regret', fontsize = 8)
  for label in (axes[idx//2, idx%2].get_xticklabels() + axes[idx//2, idx%2].get_yticklabels()):
      label.set_fontsize(6)

fig.suptitle(algo_name)
plt.show()