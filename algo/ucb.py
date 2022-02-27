### Upper-Confidence-Bound Algorithm 

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123456)

def UCB(arm_means, num_arms, total_steps, delta=1e-4):
  ### Choosing the optimal arm
  optimal_arm = np.argmax(arm_means)
  
  num_iterations = 10 # number of times we perform the same experiment to reduce randomness
  
  regret = np.zeros([total_steps, num_iterations])
  
  for iter in range(num_iterations):
    ucb = 100 * np.ones(num_arms)
    emp_means = np.zeros(num_arms)
    num_pulls = np.zeros(num_arms)
    for step_count in range(total_steps):
        greedy_arm = np.argmax(ucb)
        # generate bernoulli reward from the picked greedy arm
        reward = np.random.binomial(1, arm_means[greedy_arm])
        num_pulls[greedy_arm] += 1
        regret[step_count, iter] = arm_means[optimal_arm] - arm_means[greedy_arm]
        emp_means[greedy_arm] += (reward - emp_means[greedy_arm])/num_pulls[greedy_arm]
        ucb[greedy_arm] = emp_means[greedy_arm] + np.sqrt(2 * np.log(1/delta) / num_pulls[greedy_arm])
  
  return regret
