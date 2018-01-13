"""
    CART implementation with min_effect_size
"""
import numpy as np
import pandas as pd
import copy
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from scipy import stats
from slice_finder import *
from risk_control import *

class DecisionTree:
    
    def __init__(self, data, model):
        self.data = data
        self.model = model

        self.columns = list(data[0].columns.values)

        self.sf = SliceFinder(self.model, data)
        metrics_all = self.sf.evaluate_model(data, metric=log_loss)
        self.reference = (np.mean(metrics_all), np.std(metrics_all), len(metrics_all))

    def fit(self, max_depth=3, min_size=10):
        root = self.get_split_(self.data)
        print ( 'test', root)
        root = self.split_(root, 0, max_depth, min_size) 
        self.root = root
        return self

    def split_(self, node, depth, max_depth, min_size):
        
        # check for no split
        if node.left_group.empty or node.right_group.empty:
            return node
        
        # check for max depth
        if depth >= max_depth:
            return node

        X_left, y_left = self.data[0].loc[node.left_group], self.data[1].loc[node.left_group]
        X_right, y_right = self.data[0].loc[node.right_group], self.data[1].loc[node.right_group]

        # process left child
        if len(X_left) >= min_size:
            node.left_child = self.get_split_((X_left, y_left))
            if node.left_child is not None:
                node.left_child.parent = node
                node.left_child = self.split_(node.left_child, depth+1, max_depth, min_size)
        # process right child
        if len(X_right) >= min_size:
            node.right_child = self.get_split_((X_right, y_right))
            if node.right_child is not None:
                node.right_child.parent = node
                node.right_child = self.split_(node.right_child, depth+1, max_depth, min_size)

        return node

    def test_split_(self, X, y, attr_idx, value):
        left = X[X.iloc[:, attr_idx] < value].index
        right = X[X.iloc[:, attr_idx] >= value].index
        IG = self.entropy_(y) - len(left)/len(y) * self.entropy_(y[left]) - len(right)/len(y) * self.entropy_(y[right])
        return IG, left, right

    def get_split_(self, data):
        X, y = data[0], data[1]
        n_examples, n_features = data[0].shape
        IG, left_group, right_group, attr_idx, value  = 0, pd.DataFrame().index, pd.DataFrame().index, None, None
        for attr_idx_ in range(n_features): 
            for v in np.unique(X.iloc[:,attr_idx_]):
                IG_, left_, right_ = self.test_split_(X, y, attr_idx_, v)
                if IG < IG_:
                    IG, left_group, right_group, attr_idx, value = IG_, left_, right_, attr_idx_, v

        if attr_idx is None:
            return None
        else:
            node = Node((self.columns[attr_idx], value), left_group, right_group) 

        return node

    def entropy_(self, y):
        size = len(y)
        classes = np.unique(y)
        entropy = 0.
        for c in classes:
            p = float(np.sum(y == c)) / size             
            entropy += -p * np.log2(p)
        return entropy


    def recommend_slices(self, k=20, min_effect_size=0.3):
        recommendations = []
        candidates = [self.root]
        k_ = 0
        while len(candidates) > 0 and k_ < k:
            candidate = candidates.pop(0)
            indices = candidate.left_group.union(candidate.right_group)
            metrics = self.sf.evaluate_model((self.data[0].loc[indices], self.data[1].loc[indices]))
            eff_size = effect_size(metrics, self.reference)
            if eff_size > min_effect_size:
                candidate.size = len(indices)
                candidate.eff_size = eff_size
                recommendations.append(candidate)
                k_ += 1

            # breadth first search (prefer more interpretable slices)
            if candidate.left_child is not None:
                candidates.append(candidate.left_child)
            if candidate.right_child is not None:
                candidates.append(candidate.right_child)
            
        return recommendations

    def traverse_(self, node):
        print(node.__str__())
        if node.left_child is not None:
            self.traverse_(node.left_child)
        if node.right_child is not None:
            self.traverse_(node.right_child)

    def __str__(self):
        self.traverse_(self.root)
        return '=End-of-traverse='

class Node:
    def __init__(self, desc, left_group, right_group):
        self.desc = desc
        self.left_group = left_group
        self.right_group = right_group
        self.left_child = None
        self.right_child = None
        self.parent = None
        
    def __str__(self):
        description_ = ''
        if self.parent is None:
            description_ = 'root'
        elif self.parent.left_child is self:
            description_ = '%s < %s'%(self.desc[0], self.desc[1])
        elif self.parent.right_child is self:
            description_ = '%s >= %s'%(self.desc[0], self.desc[1])
        return '%s, size: %s'%(description_, len(self.left_group)+len(self.right_group))

    def __ancestry__(self):
        ancestors = []
        if self.parent is not None:
            ancestors.append(self.parent.__str__())
            ancestors = self.parent.__ancestry__() + ancestors
        return ancestors
    
