### KL-Upper-Confidence-Bound Algorithm 

import numpy as np
import matplotlib.pyplot as plt

DIV_MAX = 10000

def kl_divergence(p, q):
    if q == 0 and p == 0:
        return 0
    elif q == 0 and not p == 0:
        return DIV_MAX
    elif q == 1 and p == 1:
        return 0
    elif q == 1 and not p == 1:
        return DIV_MAX
    elif p == 0:
        return np.log(1/(1-q))
    elif p == 1:
        return np.log(1/q)
    return p * np.log(p/q) + (1-p) * np.log((1-p)/(1-q))

def kl_confidence(t, emp_mean, num_pulls, precision = 1e-5, max_iter = 50):
    n = 0
    lower_bound = emp_mean
    upper_bound = 1
    while n < max_iter and upper_bound - lower_bound > precision:
        q = (lower_bound + upper_bound) / 2
        if kl_divergence(emp_mean, q) > (np.log(1 + t * np.log(t) ** 2)/num_pulls):
            upper_bound = q
        else:
            lower_bound = q
        n += 1
    return (lower_bound + upper_bound)/2.

def KLUCB(arm_means, num_arms, total_steps):
  ### Choosing the optimal arm
  optimal_arm = np.argmax(arm_means)

  num_iterations = 10 # number of times we perform the same experiment
  
  regret = np.zeros([total_steps, num_iterations])
  
  for iter in range(num_iterations):
    emp_means = np.zeros(num_arms)
    num_pulls = np.zeros(num_arms)
    t = 0
    for step_count in range(0, total_steps):
        t += 1
        if step_count < num_arms:
            greedy_arm = step_count % num_arms
        else:
            # pick the best arm according to KL-UCB algorithm
            arm_confidence = np.zeros(num_arms)
            for idx in range(num_arms):
                arm_confidence[idx] = kl_confidence(t, emp_means[idx], num_pulls[idx])
            greedy_arm = np.argmax(arm_confidence)
        # generate bernoulli reward from the picked greedy arm
        reward = np.random.binomial(1, arm_means[greedy_arm])
        num_pulls[greedy_arm] += 1
        regret[step_count, iter] += arm_means[optimal_arm] - arm_means[greedy_arm]
        emp_means[greedy_arm] += (reward - emp_means[greedy_arm])/num_pulls[greedy_arm]

  return regret
