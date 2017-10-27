"""
    SliceFinder: automatic data slicing tool.

    The goal is to identify large slices that are both significant and
    interesting (e.g., high concentration of errorneous examples) for
    a given model. SliceFinder can be used to validate and debug models 
    and data. 

    Author: Yeounoh Chung (yeounohster@gmail.com)
"""

import numpy as np

class Slice:
    def __init__(self):
        pass

class SliceFinder:
    def __init__(self):
        pass

    def slicing(self, X, y):
        ''' Generate base slices '''

        for col in X.columns:
            print np.unique(X[col])
        

    def filter_by_effect_size(self, slices, epsilon):
        ''' Filter slices by the minimum effect size '''
        pass

    def alpha_investing(self, slices, alpha):
        ''' False discovery risk control '''
        pass
    
    def merge_slices(self, slices, epsilon):
        ''' Merge slices with the same filter attributes
            if the minimum effect size condition is satisfied '''
        pass


    
