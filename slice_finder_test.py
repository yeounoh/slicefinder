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
from sklearn import tree
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from slice_finder import *
from risk_control import *

from decision_tree import DecisionTree

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
        #print(column, le.classes_, le.transform(le.classes_))

X, y = adult_data[adult_data.columns.difference(["Target"])], adult_data["Target"]

# Train a model
#lr = MLPClassifier()
#lr.fit(X, y)
lr = LogisticRegression()
lr.fit(X, y)


class test_data_properties(unittest.TestCase):
    
    def test_explore_data(self):
        y_pred = lr.predict_proba(X)
        print (metrics.roc_auc_score(y.as_matrix(), y_pred[:,1]))
        y_pred_m = lr.predict_proba(X[X["Sex"] == 0])
        print (metrics.roc_auc_score(y[X["Sex"] == 0].as_matrix(), y_pred_m[:,1]))
        y_pred_f = lr.predict_proba(X[X["Sex"] == 1])
        print (metrics.roc_auc_score(y[X["Sex"] == 1].as_matrix(), y_pred_f[:,1]))
        y_pred_m = lr.predict_proba(X[X["Hours per week"] <= 40])
        print (metrics.roc_auc_score(y[X["Hours per week"] <= 40].as_matrix(), y_pred_m[:,1]))
        y_pred_f = lr.predict_proba(X[X["Hours per week"] > 40])
        print (metrics.roc_auc_score(y[X["Hours per week"] > 40].as_matrix(), y_pred_f[:,1]))
        
        sf = SliceFinder(lr, (X, y))
        metric = metrics.log_loss
        #metric = metrics.accuracy_score
        metrics_all = sf.evaluate_model((X, y),metric=metric)
        reference = (np.mean(metrics_all), np.std(metrics_all), len(metrics_all))
        metrics_female = sf.evaluate_model((X[X["Sex"] == 0], y[X["Sex"] == 0]), metric=metric)
        metrics_male = sf.evaluate_model((X[X["Sex"] == 1], y[X["Sex"] == 1]), metric=metric)
        metrics_ot = sf.evaluate_model((X[X["Hours per week"] > 40], y[X["Hours per week"] > 40]), metric=metric)
        metrics_ot_ = sf.evaluate_model((X[X["Hours per week"] <= 40], y[X["Hours per week"] <= 40]), metric=metric)
        metrics_edu = sf.evaluate_model((X[X["Education-Num"] >= 13], y[X["Education-Num"] >= 13]), metric=metric)
        
        print ("All, log_loss:", np.mean(metrics_all))
        print ("gender=Male, log_loss:", np.mean(metrics_male), 'eff size:', effect_size(metrics_male, reference))
        print ("gender=Female, log_loss:", np.mean(metrics_female), 'eff size:', effect_size(metrics_female, reference))
        print ("hours/wk > 40, log_loss:", np.mean(metrics_ot), 'eff size:', effect_size(metrics_ot, reference))
        print ("hours/wk <= 40, log_loss:", np.mean(metrics_ot_),'eff size:', effect_size(metrics_ot_, reference))
    
    def test_clustering_example(self):
        # Scale features
        scaler = StandardScaler()
        numeric_cols = ["Capital Gain", "Age", "fnlwgt", "Education-Num", "Capital Loss"]
        X_ = copy.deepcopy(X)
        X_[numeric_cols] = scaler.fit_transform(X_[numeric_cols])

        y_pred = lr.predict(X) 
        X_mis, y_mis = X_[y != y_pred], y[y != y_pred]
        reduced_data = PCA(n_components=2).fit_transform(X_mis)

        kmeans = KMeans(init='k-means++', n_clusters=20, n_init=10)
        kmeans.fit(reduced_data)

        print ('Cluster centroids')
        print ('==============================')
        print (kmeans.cluster_centers_)
        print ('==============================')
        
        x_min, x_max = reduced_data[:,0].min() - 1, reduced_data[:,0].max() + 1
        y_min, y_max = reduced_data[:,1].min() - 1, reduced_data[:,1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max-x_min)/10000), np.arange(y_min, y_max, (y_max-y_min)/10000))
        print ('figure axes')
        print ('==============================')
        print (xx.min(), xx.max(), yy.min(), yy.max())
        print ('==============================')

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


    def test_clustering(self):
        sf = SliceFinder(lr, (X,y))
        metrics_all = sf.evaluate_model((X, y))
        reference = (np.mean(metrics_all), np.std(metrics_all), len(metrics_all))

        # Scale features
        scaler = StandardScaler()
        numeric_cols = ["Capital Gain", "Age", "fnlwgt", "Education-Num", "Capital Loss"]
        X_ = copy.deepcopy(X)
        X_[numeric_cols] = scaler.fit_transform(X_[numeric_cols])

        y_pred = lr.predict(X) 
        X_mis, y_mis = X_[y != y_pred], y[y != y_pred]
        reduced_data_train = PCA(n_components=2).fit_transform(X_mis)
        reduced_data_test = PCA(n_components=2).fit_transform(X_)

        for n_clusters in range(1, 21):
            kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
            kmeans.fit(reduced_data_train) # train with mis-classified data
            y_predicted = kmeans.predict(reduced_data_test) # classify all data

            # cluster the original data with the leanred centroids
            # analyze each cluster (e.g., compute effect size)
            print('==========================')
            print('n_clusters: %s'%n_clusters)
            print('==========================')
            for cluster_id in np.unique(kmeans.labels_):
                X_cluster = X[np.array(y_predicted) == cluster_id]
                y_cluster = y[np.array(y_predicted) == cluster_id]
                eff_size_ = effect_size(sf.evaluate_model((X_cluster, y_cluster)), reference)
                print('id: %s, size: %s, eff_size: %s'%(cluster_id, len(X_cluster), eff_size_))
            print('==========================')

        
        

    def test_decision_tree(self):
        # 0- correct, 1- wrongly classified
        decisions = []
        for x_, y_ in zip(X.as_matrix(), y.as_matrix()):
            y_p = lr.predict([x_])
            if y_p == y_:
                decisions.append(0)
            else:
                decisions.append(1)
        clf = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=5000, criterion='entropy')
        clf = clf.fit(X, decisions) 
        print(X.columns)
        print(clf.score(X, decisions))
        tree.export_graphviz(clf, out_file='tree.dot',feature_names=X.columns,
                            class_names=['Correct', 'Error'])

    def test_decision_tree_with_effect_size(self):
        # 0- correct, 1- wrongly classified
        decisions = []
        for x_, y_ in zip(X.as_matrix(), y.as_matrix()):
            y_p = lr.predict([x_])
            if y_p == y_:
                decisions.append(0)
            else:
                decisions.append(1)
        
        clf = DecisionTree((X, y), lr)
        clf = clf.fit(max_depth=3, min_size=100)

        recommendations = clf.recommend_slices(k=20, min_effect_size=0.3)
        for r in recommendations:
            print ('====================')
            print (', '.join(r.__ancestry__()))
            print ('%s, %s'%(r.__str__(), r.eff_size))

class test_slice_finder(unittest.TestCase):

    def test_t_test(self):
        sf = SliceFinder(lr)
        metrics_all = sf.evaluate_model((X, y))

        reference = (np.mean(metrics_all), np.std(metrics_all), len(metrics_all))
        base_slices = sf.slicing(X, y)
        for s in base_slices:
            m_slice = sf.evaluate_model(s.data)
            print (s.__str__(), t_testing(m_slice, reference))
            
    def test_find_slice(self):
        sf = SliceFinder(lr, (X, y))
        recommendations = sf.find_slice(k=5, degree=3)
        
        for s in recommendations:
            print ('\n=====================\nSlice description:')
            for k, v in list(s.filters.items()):
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
                print ('%s:%s'%(k, values))
            print ('---------------------\neffect_size: %s'%(s.effect_size))
            print ('size: %s'%(s.size))


if __name__ == '__main__':
    unittest.main()
