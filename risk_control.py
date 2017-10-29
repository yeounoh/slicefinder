"""
    Statistical significance testing & false discovery control

    Author: Yeounoh Chung (yeounohster@gmail.com)
"""
from scipy import stats
import numpy as np



def t_testing(sample_a, sample_b, alpha=0.05):
    t, prob = stats.ttest_ind(sample_a, sample_b)
    return prob <= alpha

def t_testing(sample_a, alpha=0.05):
    pass

def effect_size(sample_a, sample_b):
    diff = np.mean(sample_a) - np.mean(sample_b) 
    diff /= (np.std(sample_a)+np.std(sample_b))/2.
    return diff

def effect_size(sample_a, reference):
    mu, s, n = reference[0], reference[1], reference[2]
    sample_b_mean = (mu*n - np.sum(sample_a))/(n-len(sample_a))
    sample_b_std = (s**2*(n-1) - np.std(sample_a)**2*(len(sample_a)-1))/(n-len(sample_a)-1)

    diff = np.mean(sample_a) - sample_b_mean
    diff /= (np.std(sample_a) + sample_b_std)/2.
    return diff
