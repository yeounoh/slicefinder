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
from sklearn.metrics import log_loss
from scipy import stats
from risk_control import *

"""
    Slice is specified with a dictionary that maps a set of attributes 
    and their values. For instance, '1 <= age < 5' is expressed as {'age':[1,5]}
    and 'gender = male' as {'gender':['male']}
"""
class Slice:
    def __init__(self, filters, data, complement):
        self.filters = filters
        self.data = data
        self.complement = complement

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
    def __init__(self, model):
        self.model = model

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
                    data = (X[ np.logical_and(bin_edges[i] <= X[col], X[col] < bin_edges[i+1]) ],
                               y[ np.logical_and(bin_edges[i] <= X[col], X[col] < bin_edges[i+1]) ] ) 
                    complement = (X[ np.logical_or(bin_edges[i] > X[col], X[col] >= bin_edges[i+1]) ],
                               y[ np.logical_or(bin_edges[i] > X[col], X[col] >= bin_edges[i+1]) ] )
                    s = Slice({col:[bin_edges[i],bin_edges[i+1]]}, data, complement)
                    base_slices.append(s)
            else:
                for v in uniques:
                    data = (X[X[col] == v], y[X[col] == v])
                    complement = (X[X[col] != v], y[X[col] != v])
                    s = Slice({col:[v]}, data, complement)                 
                    base_slices.append(s)

        return base_slices

    def evaluate_model(self, data, labels=[ 0, 1 ], metric=log_loss):
        X, y = data[0].as_matrix(), data[1].as_matrix()

        metric_by_example = []
        for x_, y_ in zip(X, y):
            y_p = self.model.predict_proba([x_])
            metric_by_example.append(metric([y_], y_p, labels=labels))

        return metric_by_example
        
    def cross_slice(self, slices):
        pass

    def filter_by_effect_size(self, slices, reference, epsilon=0.5):
        ''' Filter slices by the minimum effect size '''
        filtered_slices = []
        for s in slices:
            m_slice = self.evaluate_model(s.data)
            eff_size = effect_size(m_slice, reference)
            if eff_size >= epsilon:
                filtered_slices.append(s)
        return filtered_slices

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
