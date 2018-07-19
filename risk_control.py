"""
    Statistical significance testing & false discovery control

    Author: Yeounoh Chung (yeounohster@gmail.com)
"""
from scipy import stats
import numpy as np
import math


def t_testing(sample_a, reference, alpha=0.05):
    ''' Unpaired two-sample (Welch's) t-test '''
    mu, s, n = reference[0], reference[1], reference[2]
    sample_b_mean = (mu*n - np.sum(sample_a))/(n-len(sample_a))
    sample_b_var = (s**2*(n-1) - np.std(sample_a)**2*(len(sample_a)-1))/(n-len(sample_a)-1)

    t = np.mean(sample_a) - sample_b_mean
    t /= math.sqrt( np.var(sample_a)/len(sample_a) + sample_b_var/(n-len(sample_a)) )

    prob = stats.norm.cdf(t)
    return prob
    

def effect_size(sample_a, reference):
    mu, s, n = reference[0], reference[1], reference[2]
    if n-len(sample_a) == 0:
        return 0
    sample_b_mean = (mu*n - np.sum(sample_a))/(n-len(sample_a))
    sample_b_var = (s**2*(n-1) - np.std(sample_a)**2*(len(sample_a)-1))/(n-len(sample_a)-1)
    if sample_b_var < 0:
        sample_b_var = 0.

    diff = np.mean(sample_a) - sample_b_mean
    diff /= math.sqrt( (np.std(sample_a) + math.sqrt(sample_b_var))/2. )
    return diff
