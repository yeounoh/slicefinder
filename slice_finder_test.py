"""
    Test SliceFinder and its slice exploration algorithm
    (Design doc at https://goo.gl/J7EJ9b)

    Author: Yeounoh Chung (yeounohster@gmail.com)
"""

import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sklearn.linear_model as linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
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


# Train a model
lr = MLPClassifier(alpha=1)
lr.fit(X, y)
#lr = LogisticRegression()
#lr.fit(X, y)


class test_data_properties(unittest.TestCase):
    
    def test_explore_data(self):
        metric = metrics.log_loss
        metric = metrics.accuracy_score
        y_pred = lr.predict_proba(X)
        print metrics.roc_auc_score(y.as_matrix(), y_pred[:,1])
        y_pred_m = lr.predict_proba(X[X["Sex"] == 0])
        print metrics.roc_auc_score(y[X["Sex"] == 0].as_matrix(), y_pred_m[:,1])
        y_pred_f = lr.predict_proba(X[X["Sex"] == 1])
        print metrics.roc_auc_score(y[X["Sex"] == 1].as_matrix(), y_pred_f[:,1])
        y_pred_m = lr.predict_proba(X[X["Hours per week"] <= 40])
        print metrics.roc_auc_score(y[X["Hours per week"] <= 40].as_matrix(), y_pred_m[:,1])
        y_pred_f = lr.predict_proba(X[X["Hours per week"] > 40])
        print metrics.roc_auc_score(y[X["Hours per week"] > 40].as_matrix(), y_pred_f[:,1])
        
        sf = SliceFinder(lr)
        metrics_all = sf.evaluate_model((X, y),metric=metric)
        metrics_male = sf.evaluate_model((X[X["Sex"] == 0], y[X["Sex"] == 0]), metric=metric)
        metrics_female = sf.evaluate_model((X[X["Sex"] == 1], y[X["Sex"] == 1]), metric=metric)
        metrics_ot = sf.evaluate_model((X[X["Hours per week"] > 40], y[X["Hours per week"] > 40]), metric=metric)
        metrics_ot_ = sf.evaluate_model((X[X["Hours per week"] <= 40], y[X["Hours per week"] <= 40]), metric=metric)
        metrics_edu = sf.evaluate_model((X[X["Education-Num"] >= 13], y[X["Education-Num"] >= 13]), metric=metric)
        print np.mean(metrics_all), np.mean(metrics_ot), np.mean(metrics_edu), np.mean(metrics_male), np.mean(metrics_female)
        print np.mean(metrics_ot_)
    
    def test_model_understanding(self):
        # Scale features
        scaler = StandardScaler()
        numeric_cols = ["Capital Gain", "Age", "fnlwgt", "Education-Num", "Capital Loss"]
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        y_pred = lr.predict(X) 
        X_mis, y_mis = X[y != y_pred], y[y != y_pred]
        reduced_data = PCA(n_components=2).fit_transform(X_mis)

        kmeans = KMeans(init='k-means++', n_clusters=20, n_init=10)
        kmeans.fit(reduced_data)
        print kmeans.cluster_centers_
        print X_mis[np.array(kmeans.labels_) == 13]
        x_min, x_max = reduced_data[:,0].min() - 1, reduced_data[:,0].max() + 1
        y_min, y_max = reduced_data[:,1].min() - 1, reduced_data[:,1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max-x_min)/10000), np.arange(y_min, y_max, (y_max-y_min)/10000))
        print xx.min(), xx.max(), yy.min(), yy.max()
        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')

        plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='w', zorder=10)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.savefig('clusters.png')

class test_slice_finder(unittest.TestCase):

    def test_t_test(self):
        sf = SliceFinder(lr)
        metrics_all = sf.evaluate_model((X, y))


        reference = (np.mean(metrics_all), np.std(metrics_all), len(metrics_all))
        base_slices = sf.slicing(X, y)
        for s in base_slices:
            m_slice = sf.evaluate_model(s.data)
            print s.__str__(), t_testing(m_slice, reference)
            
    def test_filter_by_effect_size(self):
        sf = SliceFinder(lr)
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
        sf = SliceFinder(lr, (X, y))
        recommendations = sf.find_slice(k=10)
        
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
    #suite.addTest(test_data_properties("test_explore_data"))
    #suite.addTest(test_data_properties("test_model_understanding"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
