"""
    SliceFinder: automatic data slicing tool.

    The goal is to identify large slices that are both significant and
    interesting (e.g., high concentration of errorneous examples) for
    a given model. SliceFinder can be used to validate and debug models 
    and data. 

    Author: Yeounoh Chung (yeounohster@gmail.com)
"""

import numpy as np
import copy
from scipy import stats
from risk_control import *

"""
    Slice is specified with a dictionary that maps a set of attributes 
    and their values. For instance, '1 <= age < 5' is expressed as {'age':[1,5]}
    and 'gender = male' as {'gender':['male']}
"""
class Slice:
    def __init__(self, filters, data):
        self.filters = filters
        self.data = data

    def get_filter(self):
        return self.filters

    def set_filter(self, filters):
        self.filters = filters

    def union(self, s):
        ''' union with Slice s '''
        pass

    def __str__(self):
        slice_desc = ''
        for k, v in self.filters.iteritems():
            slice_desc += '%s:%s '%(k,v)
        return slice_desc 

class SliceFinder:
    def __init__(self):
        pass

    def slicing(self, X, y):
        ''' Generate base slices '''
        n, m = X.shape[0], X.shape[1]

        base_slices = []
        for col in X.columns:
            uniques, counts = np.unique(X[col], return_counts=True)
            if len(uniques) == n:
                # Skip ID like col
                continue
            if len(uniques) > n/2.:
                # Bin high cardinality col
                bin_edges = binning(X[col], n_bin=10)
                for i in range(len(bin_edges)-1):
                    ta = bin_edges[i] <= X[col]
                    s = Slice({col:[bin_edges[i],bin_edges[i+1]]},
                              (X[ np.logical_and(bin_edges[i] <= X[col], X[col] < bin_edges[i+1]) ],
                               y[ np.logical_and(bin_edges[i] <= X[col], X[col] < bin_edges[i+1]) ] ))
                    base_slices.append(s)
            else:
                for v in uniques:
                    s = Slice({col:[v]}, (X[X[col] == v], y[X[col] == v]))                 
                    base_slices.append(s)

        return base_slices
        

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


def binning(col, n_bin=10):
    ''' Equi-height binning '''
    bin_edges = stats.mstats.mquantiles(col, np.arange(0., 1.+1./n_bin, 1./n_bin))
    return bin_edges
