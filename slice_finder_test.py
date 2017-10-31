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

# drop nan values
adult_data = adult_data.dropna()
# Encode categorical features
encoders = {}
for column in adult_data.columns:
    if adult_data.dtypes[column] == np.object:
        le = LabelEncoder()
        adult_data[column] = le.fit_transform(adult_data[column])
        encoders[column] = le

# Split data into train and test sets
X, y = adult_data[adult_data.columns.difference(["Target"])], adult_data["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, test_size=0.5)

# Scale features
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = scaler.transform(X_test)

# Train a model
mlp = MLPClassifier(alpha=1)
mlp.fit(X, y)

class test_slice_finder(unittest.TestCase):

    def test_t_test(self):
        sf = SliceFinder(mlp)
        metrics_all = sf.evaluate_model((X, y))
        reference = (np.mean(metrics_all), np.std(metrics_all), len(metrics_all))
        base_slices = sf.slicing(X, y)
        for s in base_slices:
            m_slice = sf.evaluate_model(s.data)
            print s.__str__(), t_testing(m_slice, reference)
            
    def test_filter_by_effect_size(self):
        sf = SliceFinder(mlp)
        metrics_all = sf.evaluate_model((X, y))
        reference = (np.mean(metrics_all), np.std(metrics_all), len(metrics_all))

        # degree 1
        #base_slices = sf.slicing(X, y)
        #filtered_slices1, rejected_slices1 = sf.filter_by_effect_size(base_slices, reference, epsilon=0.2)
        #pickle.dump((filtered_slices1, rejected_slices1), open('degree1.p','wb'))
        #dump1 = pickle.load(open('degree1.p','rb'))
        #filtered_slices1, rejected_slices1 = dump1[0], dump1[1]

        # degree 2
        #crossed_slices2 = sf.crossing2(rejected_slices1)
        #filtered_slices2, rejected_slices2 = sf.filter_by_effect_size(crossed_slices2, reference, epsilon=0.2)
        #pickle.dump((filtered_slices2, rejected_slices2), open('degree2.p', 'wb'))
        #dump2 = pickle.load(open('degree2.p','rb'))
        #filtered_slices2, rejected_slices2 = dump2[0], dump2[1]

        # degree 3: memory issue (kill: 9)
        #crossed_slices3 = sf.crossing3(rejected_slices2, rejected_slices1)
        #filtered_slices3, rejected_slices3 = sf.filter_by_effect_size(crossed_slices3, reference, epsilon=0.2)
        #pickle.dump((filtered_slices3, rejected_slices3), open('degree3.p', 'wb'))
        #dump3 = pickle.load(open('degree3.p','rb'))
        #filtered_slices3 = dump3[0]

    def test_alpha_investing(self):
        pass

    def test_merge_slices(self):
        pass

    def test_find_slice(self):
        sf = SliceFinder(mlp)
        recommendations = sf.find_slice(X, y, k=10)
        
        for s in recommendations:
            print '\n=====================\nSlice description:'
            for k, v in s.filters.iteritems():
                values = ''
                if k in encoders:
                    le = encoders[k]
                    for v_ in v:
                        values += '%s '%(le.inverse_transform(v_)[0])
                else:
                    for v_ in sorted(v, key=lambda x: x[0]):
                        if len(v_) > 1:
                            values += '%s ~ %s'%(v_[0], v_[1])
                        else:
                            values += '%s '%(v_[0])
                print '%s:%s'%(k, values)
            print '---------------------\neffect_size: %s'%(s.effect_size)
            print 'size: %s'%(s.size)


if __name__ == '__main__':
    #unittest.main()
    suite = unittest.TestSuite()
    suite.addTest(test_slice_finder("test_find_slice"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
