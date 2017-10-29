"""
    Statistical significance testing & false discovery control

    Author: Yeounoh Chung (yeounohster@gmail.com)
"""
from scipy import stats
import numpy as np
import math



def t_testing(sample_a, sample_b, alpha=0.05):
    t, prob = stats.ttest_ind(sample_a, sample_b)
    return prob <= alpha

def t_testing(sample_a, reference, alpha=0.05):
    mu, s, n = reference[0], reference[1], reference[2]
    sample_b_mean = (mu*n - np.sum(sample_a))/(n-len(sample_a))
    sample_b_var = (s**2*(n-1) - np.std(sample_a)**2*(len(sample_a)-1))/(n-len(sample_a)-1)
    pass

def effect_size(sample_a, reference):
    mu, s, n = reference[0], reference[1], reference[2]
    sample_b_mean = (mu*n - np.sum(sample_a))/(n-len(sample_a))
    sample_b_var = (s**2*(n-1) - np.std(sample_a)**2*(len(sample_a)-1))/(n-len(sample_a)-1)

    diff = np.mean(sample_a) - sample_b_mean
    diff /= (np.std(sample_a) + math.sqrt(sample_b_var))/2.
    return diff
