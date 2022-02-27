### Explore-then-Commit Algorithm 
### Choose each arm sequentially m times, then follow the best one estimate.

import numpy as np
import matplotlib.pyplot as plt

def ETC(arm_means, num_arms, total_steps, m):
  ### Choosing the optimal arm
  optimal_arm = np.argmax(arm_means)

  num_iterations = 10 # number of times we perform the same experiment

  regret = np.zeros([total_steps,num_iterations])
  
  for iter in range(num_iterations):
    num_steps = 0
    emp_means = np.zeros(num_arms)
    num_pulls = np.zeros(num_arms)
    # Pure Exploration Phase
    for i in range(m):
        for j in range(num_arms):
          num_pulls[j] += 1
          # generate bernoulli reward from the picked greedy arm
          reward = np.random.binomial(1, arm_means[j])
          emp_means[j] += (reward - emp_means[j])/num_pulls[j]
          regret[num_steps, iter] += arm_means[optimal_arm] - arm_means[j]
          num_steps += 1
      
    # Pure Exploitation Phase
    selected_best_arm = np.argmax(emp_means)

    exploitation_steps = total_steps - m * num_arms
    for _ in range(exploitation_steps):
      regret[num_steps, iter] += arm_means[optimal_arm] - arm_means[selected_best_arm]
      num_steps += 1
  
  return regret
