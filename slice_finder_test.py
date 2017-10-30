"""
    Test SliceFinder and its slice exploration algorithm
    (Design doc at https://goo.gl/J7EJ9b)

    Author: Yeounoh Chung (yeounohster@gmail.com)
"""

import unittest
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sklearn.linear_model as linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from slice_finder import *
from risk_control import *

adult_data = pd.read_csv(
    "data/adult.data",
    names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

adult_data = adult_data.dropna()

# Encode categorical features
encoders = {}
for column in adult_data.columns:
    if adult_data.dtypes[column] == np.object:
        le = LabelEncoder()
        adult_data[column] = le.fit_transform(adult_data[column])
        encoders[column] = le

# Split data into train and test sets
X, y = adult_data[adult_data.columns-["Target"]], adult_data["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

# Scale features
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = scaler.transform(X_test)

# Train a model
mlp = MLPClassifier(alpha=1)
mlp.fit(X, y)

class test_slice_finder(unittest.TestCase):

    def test_filter_by_effect_size(self):
        sf = SliceFinder(mlp)
        metrics_all = sf.evaluate_model((X, y))
        reference = (np.mean(metrics_all), np.std(metrics_all), len(metrics_all))

        # degree 1
        base_slices = sf.slicing(X, y)
        filtered_slices1, rejected_slices1 = sf.filter_by_effect_size(base_slices, reference, epsilon=0.2)

        # degree 2
        crossed_slices2 = sf.crossing2(rejected_slices1)
        filtered_slices2, rejected_slices2 = sf.filter_by_effect_size(crossed_slices2, reference, epsilon=0.2)

        # degree 3
        crossed_slices3 = sf.crossing3(rejected_slices2, rejected_slices1)
        filtered_slices3, rejected_slices3 = sf.filter_by_effect_size(crossed_slices3, reference, epsilon=0.2)

        print ('%s interesting slices'%(len(filtered_slices1+filtered_slices2+filtered_slices3)))

    def test_alpha_investing(self):
        pass

    def test_merge_slices(self):
        pass

if __name__ == '__main__':
    unittest.main()
