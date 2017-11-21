"""
    SliceFinder: automatic data slicing tool.

    The goal is to identify large slices that are both significant and
    interesting (e.g., high concentration of errorneous examples) for
    a given model. SliceFinder can be used to validate and debug models 
    and data. 

    Author: Yeounoh Chung (yeounohster@gmail.com)
"""

import numpy as np
import pandas as pd
import copy
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from scipy import stats
from risk_control import *

"""
    Slice is specified with a dictionary that maps a set of attributes 
    and their values. For instance, '1 <= age < 5' is expressed as {'age':[[1,5]]}
    and 'gender = male' as {'gender':[['male']]}
"""
class Slice:
    def __init__(self, filters, data):
        self.filters = filters
        self.data = data
        self.size = data[0].shape[0]
        self.effect_size = None

    def get_filter(self):
        return self.filters

    def set_filter(self, filters):
        self.filters = filters

    def set_effect_size(self, effect_size):
        self.effect_size = effect_size

    def union(self, s):
        ''' union with Slice s '''
        if set(self.filters.keys()) == set(s.filters.keys()):
            for k in self.filters.keys():
                self.filters[k] = self.filters[k] + s.filters[k]
        else:
            return False

        idx = self.data[0].index.difference(s.data[0].index)
        frames_X = [self.data[0].loc[idx], s.data[0]]
        frames_y = [self.data[1].loc[idx], s.data[1]]
        self.data = (pd.concat(frames_X), pd.concat(frames_y))
        self.size = self.data[0].shape[0]

        return True

    def intersect(self, s):
        ''' intersect with Slice s '''
        for k, v in s.filters.iteritems():
            if k not in self.filters:
                self.filters[k] = v
            else:
                for condition in s.filters[k]:
                    if condition not in self.filters[k]:
                        self.filters[k].append(condition)

        idx = self.data[0].index.intersection(s.data[0].index)
        self.data = (self.data[0].loc[idx], self.data[1].loc[idx])
        self.size = self.data[0].shape[0]

        return True

    def __str__(self):
        slice_desc = ''
        for k, v in self.filters.iteritems():
            slice_desc += '%s:%s '%(k,v)
        return slice_desc 

class SliceFinder:
    def __init__(self, model):
        self.model = model

    def find_slice(self, X, y, k=50, epsilon=0.2, alpha=0.05, degree=2):
        ''' Find interesting slices '''
        assert k > 0, 'Number of recommendation k should be greater than 0'

        metrics_all = self.evaluate_model((X, y))
        reference = (np.mean(metrics_all), np.std(metrics_all), len(metrics_all))

        slices = []
        for i in range(1,degree+1):
            # degree 1~3 feature crosses
            if i == 1:
                candidates = self.slicing(X, y)
            elif i == 2:
                candidates = self.crossing2(not_interesting)
            elif i == 3:
                candidates = self.crossing3(not_interesting)
            interesting, not_interesting = self.filter_by_effect_size(candidates, reference, epsilon)
            slices += interesting
    
            slices = self.merge_slices(slices, reference, epsilon)

            slices, rejected = self.filter_by_significance(slices, reference, alpha)    

            if len(slices) >= k:
                break

        return slices[:k]
            
    def slicing(self, X, y):
        ''' Generate base slices '''
        n, m = X.shape[0], X.shape[1]

        slices = []
        for col in X.columns:
            uniques, counts = np.unique(X[col], return_counts=True)
            if len(uniques) == n:
                # Skip ID like col
                continue
            if len(uniques) > n/2.:
                # Bin high cardinality col
                bin_edges = self.binning(X[col], n_bin=10)
                for i in range(len(bin_edges)-1):
                    data = (X[ np.logical_and(bin_edges[i] <= X[col], X[col] < bin_edges[i+1]) ],
                               y[ np.logical_and(bin_edges[i] <= X[col], X[col] < bin_edges[i+1]) ] ) 
                    s = Slice({col:[[bin_edges[i],bin_edges[i+1]]]}, data)
                    slices.append(s)
            else:
                for v in uniques:
                    data = (X[X[col] == v], y[X[col] == v])
                    s = Slice({col:[[v]]}, data)                 
                    slices.append(s)

        return slices

    def crossing2(self, slices):
        ''' Cross uninteresting base slices together '''
        crossed_slices = []
        for i in range(len(slices)-1):
            for j in range(i+1, len(slices)):
                slice_ij = copy.deepcopy(slices[i])
                slice_ij.intersect(slices[j])
                crossed_slices.append(slice_ij)

        return crossed_slices

    def crossing3(self, slices2, slices1):
        ''' Cross uninteresting 2-degree cross and base slices together '''
        crossed_slices = []
        for s2 in slices2:
            for s1 in slices1:
                for k, v in s1.filters.iteritems():
                    if k in s2.filters and v[0] in s2.filters[k]:
                        continue
                    
                    slice_ijk = copy.deepcopy(s2)
                    slice_ijk.intersect(s1)
                    crossed_slices.append(slice_ijk)
                        
        return crossed_slices

    def evaluate_model(self, data, metric=log_loss):
        ''' evaluate model on a given data (X, y), example by example '''
        X, y = data[0].as_matrix(), data[1].as_matrix()
        
        metric_by_example = []
        for x_, y_ in zip(X, y):
            if metric == log_loss:
                y_p = self.model.predict_proba([x_])
                metric_by_example.append(metric([y_], y_p, labels=self.model.classes_))
            elif metric == accuracy_score:
                y_p = self.model.predict([x_])
                metric_by_example.append(metric([y_], y_p))

        return metric_by_example
        
    def filter_by_effect_size(self, slices, reference, epsilon=0.5):
        ''' Filter slices by the minimum effect size '''
        filtered_slices = []
        rejected = []
        for s in slices:
            if s.size == 0:
                continue

            m_slice = self.evaluate_model(s.data)
            eff_size = effect_size(m_slice, reference)
            s.set_effect_size(eff_size) # Update effect size
            if eff_size >= epsilon:
                filtered_slices.append(s)
            else:
                rejected.append(s)
        return filtered_slices, rejected
    
    def merge_slices(self, slices, reference, epsilon):
        ''' Merge slices with the same filter attributes
            if the minimum effect size condition is satisfied '''
        merged_slices = []

        sorted_slices = sorted(slices, key=lambda x: x.effect_size, reverse=True)
        taken = []
        for i in range(len(sorted_slices)-1):
            if i in taken: continue

            s_ = copy.deepcopy(sorted_slices[i])
            taken.append(i)
            for j in range(i, len(sorted_slices)):
                if j in taken: continue

                prev = copy.deepcopy(s_)
                if s_.union(sorted_slices[j]):
                    m_s_ = self.evaluate_model(s_.data)
                    eff_size = effect_size(m_s_, reference)
                    if eff_size >= epsilon:
                        s_.set_effect_size(eff_size)
                        taken.append(j)
                    else:
                        s_ = prev

            merged_slices.append(s_)

        return merged_slices

    def filter_by_significance(self, slices, reference, alpha):
        ''' Return significant slices '''
        filtered_slices = []
        rejected = []
        for s in slices:
            if s.size == 0:
                continue

            m_slice = self.evaluate_model(s.data)
            if t_testing(m_slice, reference, alpha):
                filtered_slices.append(s)
            else:
                rejected.append(s)
        return filtered_slices, rejected
        

    def binning(self, col, n_bin=10):
        ''' Equi-height binning '''
        bin_edges = stats.mstats.mquantiles(col, np.arange(0., 1.+1./n_bin, 1./n_bin))
        return bin_edges
