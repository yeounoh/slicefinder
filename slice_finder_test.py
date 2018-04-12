"""
Test SliceFinder and its slice exploration algorithm
    (Design doc at https://goo.gl/J7EJ9b)

    Author: Yeounoh Chung (yeounohster@gmail.com)
"""

import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 17})
import pickle
import random
import concurrent.futures
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sklearn.linear_model as linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier

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
        print(column, le.classes_, le.transform(le.classes_))

X, y = adult_data[adult_data.columns.difference(["Target"])], adult_data["Target"]
print ( X.columns)
# Train a model
lr = LogisticRegression()
lr.fit(X, y)
rf = RandomForestClassifier(max_depth=5, n_estimators=10)
rf.fit(X, y)
lr = rf


# s_(a, y, y^)


class corrected_model:
    def __init__(self, model, p):
        self.model = model
        self.p = p
        self.classes_ = [0, 1]

    def predict_proba(self, X):
        result = list()
        for x in X:
            proba = self.model.predict_proba([x])
            proba0, proba1 = proba[0][0] , proba[0][1]
            y_hat = int(proba1 > proba0)
            A = x[11]
            if A == 0 and y_hat == 1:
                result.append([1-self.p[1], self.p[1]])
            elif A == 0 and y_hat == 0:
                result.append([1-self.p[0], self.p[0]])
            elif A == 1 and y_hat == 1:
                result.append([1-self.p[3], self.p[3]])
            elif A == 1 and y_hat == 0:
                result.append([1-self.p[2], self.p[2]])
        return np.array(result)
        

class test_data_properties(unittest.TestCase):

    def test_multi_hypotheses(self):
        dataset = 'fraud'
        encoders = {}

        '''
        main_df = pd.read_csv("data/creditcard.csv")
        main_df = main_df.dropna()
        main_df.head()
        feature_size = len(main_df.columns)
        class_index = feature_size -1 

        fraud_raw = main_df[main_df['Class'] == 1]
        normal_raw = main_df[main_df['Class'] == 0]

        # Undersample the normal transactions
        percentage = len(fraud_raw)/float(len(normal_raw))
        normal = normal_raw.sample(frac=percentage)
        fraud = fraud_raw
        cc_data = fraud.append(normal)

        X, y = cc_data[cc_data.columns.difference(["Class"])], cc_data["Class"]

        reg_model = RandomForestClassifier(criterion='entropy',n_estimators=100)
        lr = reg_model.fit(X,y)
        '''
        '''
        sf = SliceFinder(lr, (X, y))
        metrics_all = sf.evaluate_model((X,y))
        reference = (np.mean(metrics_all), np.std(metrics_all), len(metrics_all))
    
        recommendations = sf.find_slice(k=100, epsilon=0.4, degree=10, risk_control=False)
        slices, uninteresting = list(), list()
        with open('slices.p','rb') as handle:
            slices = pickle.load(handle)
        with open('uninteresting.p', 'rb') as handle:
            uninteresting = pickle.load(handle)  
        slices, rejected = sf.filter_by_significance(slices, reference, 0.05)
        for s in rejected:
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
            print ('---------------------\nmetric: %s'%(s.metric))
            print ('size: %s'%(s.size))

        print (len(slices), len(rejected)) # fraud (1544, 4182), census (2530, 876)
        print (np.mean([s.size for s in slices]), np.mean([s.size for s in rejected])) # fraud (7.86, 8.36), census (44.19, 5.18)
        print (np.mean([s.effect_size for s in slices]), np.mean([s.effect_size for s in rejected])) # fraud (1.54, 0.60), census (1.28, 0.65)       
        with open('accepted_%s.p'%dataset,'wb') as handle:
            pickle.dump(slices, handle)
        with open('rejected_%s.p'%dataset, 'wb') as handle:
            pickle.dump(rejected, handle)
        '''
        dataset='fraud'
        with open('accepted_%s.p'%dataset,'rb') as handle:
            slices = pickle.load(handle)
        with open('rejected_%s.p'%dataset, 'rb') as handle:
            rejected = pickle.load(handle)  
        plt.figure(1, figsize=(6,5))
        significant = [s.size for s in slices if s.size]
        insignificant = [s.size for s in rejected if s.size] 
        bins = np.linspace(0, max(significant), 50)
        plt.hist(significant, bins, alpha=0.5, label='Accepted')
        plt.hist(insignificant, bins, alpha=0.5, label='Rejected')
        plt.xlim([0,100])
        plt.legend(loc='upper right')
        plt.xlabel('Slice Size',fontsize=17)
        plt.tight_layout()
        plt.savefig('longtail_%s.pdf'%dataset)

        
        
        

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

        metrics_bachelors = sf.evaluate_model((X[X["Education-Num"] == 13], y[X["Education-Num"] == 13]), metric=metric)
        metrics_masters = sf.evaluate_model((X[X["Education-Num"] == 14], y[X["Education-Num"] == 14]), metric=metric)
        metrics_doctorate = sf.evaluate_model((X[X["Education-Num"] == 15], y[X["Education-Num"] == 15]), metric=metric)
        metrics_preschool = sf.evaluate_model((X[X["Education"] == 13], y[X["Education"] == 13]), metric=metric)
        metrics_hsgrad = sf.evaluate_model((X[X["Education"] == 11], y[X["Education"] == 11]), metric=metric)
       
        
        metrics_white = sf.evaluate_model((X[X["Race"] == 4], y[X["Race"] == 4]), metric=metric)
        metrics_black = sf.evaluate_model((X[X["Race"] == 2], y[X["Race"] == 2]), metric=metric)
        metrics_asian = sf.evaluate_model((X[X["Race"] == 1], y[X["Race"] == 1]), metric=metric)
        metrics_indian = sf.evaluate_model((X[X["Race"] == 0], y[X["Race"] == 0]), metric=metric)
        metrics_others= sf.evaluate_model((X[X["Race"] == 3], y[X["Race"] == 3]), metric=metric)
        
        
        metrics_workclass= sf.evaluate_model((X[(X["Workclass"] == 1) & (X["Race"] == 4)], y[(X["Workclass"] == 1) & (X["Race"] == 4)]), metric=metric)

        print ("All, log_loss:", np.mean(metrics_all))
        print ("gender=Male, log_loss:", np.mean(metrics_male), 'eff size:', effect_size(metrics_male, reference))
        print ("gender=Female, log_loss:", np.mean(metrics_female), 'eff size:', effect_size(metrics_female, reference))
        print ("hours/wk > 40, log_loss:", np.mean(metrics_ot), 'eff size:', effect_size(metrics_ot, reference))
        print ("hours/wk <= 40, log_loss:", np.mean(metrics_ot_),'eff size:', effect_size(metrics_ot_, reference))
        print ("race=white, log_loss:", np.mean(metrics_white),'eff size:', effect_size(metrics_white, reference))
        print ("race=black, log_loss:", np.mean(metrics_black),'eff size:', effect_size(metrics_black, reference))
        print ("race=asian, log_loss:", np.mean(metrics_asian),'eff size:', effect_size(metrics_asian, reference))
        print ("race=indian, log_loss:", np.mean(metrics_indian),'eff size:', effect_size(metrics_indian, reference))
        print ("race=others, log_loss:", np.mean(metrics_others),'eff size:', effect_size(metrics_others, reference))
        print ("education=doctorate, log_loss:", np.mean(metrics_doctorate),'eff size:', effect_size(metrics_doctorate, reference))
        print (len(X[X["Education-Num"] == 13]))
        print ("education=masters, log_loss:", np.mean(metrics_masters),'eff size:', effect_size(metrics_masters, reference))
        print (len(X[X["Education-Num"] == 14]))
        print ("education=bachelors, log_loss:", np.mean(metrics_bachelors),'eff size:', effect_size(metrics_bachelors, reference))
        print (len(X[X["Education-Num"] == 15]))
        print ("education=preschool, log_loss:", np.mean(metrics_preschool),'eff size:', effect_size(metrics_preschool, reference))
        print (len(X[X["Education"] == 13]))
        print ("education=hs, log_loss:", np.mean(metrics_hsgrad),'eff size:', effect_size(metrics_hsgrad, reference))
        print (len(X[X["Education"] == 11]))
        print ("workclass=local-gov and race=white, log_loss:", np.mean(metrics_workclass),'eff size:', effect_size(metrics_workclass, reference))
    
    def test_model_fairness(self):
        from sklearn.metrics import roc_curve, auc

        
        plt.figure()
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        y_pred = rf.predict_proba(X)[:,1]
        fpr, tpr, _ = roc_curve(y.as_matrix().ravel(), y_pred.ravel())
        plt.plot(fpr, tpr, lw=2,label='RF')
        y_pred = lr.predict_proba(X)[:,1]
        fpr, tpr, _ = roc_curve(y.as_matrix().ravel(), y_pred.ravel())
        plt.plot(fpr, tpr, lw=2,label='LR')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig('roc.png')

        plt.figure()
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        labels = ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other Minotiries', 'White'] 
        auc_ = list()
        for i in range(len(labels)):
            y_pred = rf.predict_proba(X[X['Race'] == i])[:,1]
            fpr, tpr, _ = roc_curve(y[X['Race'] == i].as_matrix().ravel(), y_pred.ravel(), drop_intermediate=False)
            auc_.append(auc(fpr, tpr)) 
            plt.plot(fpr, tpr, lw=2,label=labels[i])
        print(labels)
        print(auc_) 

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig('roc_race.png')
        """
        plt.figure()
        ax = plt.gca()
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        labels = ['10th', '11th' '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm',
 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool',
 'Prof-school', 'Some-college'] 
        for i in range(len(labels)):
            y_pred = rf.predict_proba(X[X['Education'] == i])[:,1]
            fpr, tpr, _ = roc_curve(y[X['Education'] == i].as_matrix().ravel(), y_pred.ravel())
            plt.plot(fpr, tpr, lw=2,label=labels[i])

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig('roc_education.png')
        """
        plt.figure()
        ax = plt.gca()
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        labels = ['Female', 'Male'] 
        auc_ = list()
        for i in range(len(labels)):
            y_pred = rf.predict_proba(X[X['Sex'] == i])[:,1]
            fpr, tpr, _ = roc_curve(y[X['Sex'] == i].as_matrix().ravel(), y_pred.ravel())
            auc_.append(auc(fpr, tpr)) 
            plt.plot(fpr, tpr, lw=2,label=labels[i]+' (auc:%.2f'%(auc(fpr,tpr))+')')
        print(labels)
        print(auc_) 

        # linear programming to correct the classifier
        from scipy.optimize import linprog, minimize

        s_000, s_001, s_010, s_011, s_100, s_101, s_110, s_111 = 0, 0, 0, 0, 0, 0, 0, 0
        for idx, row in X.iterrows():
            A = row['Sex']
            y_orig = y.loc[idx]
            y_hat = rf.predict(row.reshape(-1,1).T)

            if A == 0:
                if y_orig == 0:
                    if y_hat == 0:
                        s_000 += 1
                    else:
                        s_001 += 1
                else:
                    if y_hat == 0:
                        s_010 += 1
                    else:
                        s_011 += 1
            else:
                if y_orig == 0:
                    if y_hat == 0:
                        s_100 += 1
                    else:
                        s_101 += 1
                else:
                    if y_hat == 0:
                        s_110 += 1
                    else:
                        s_111 += 1
        c = [1./X.size * (s_000 - s_010), 
             1./X.size * (s_001 - s_011),
             1./X.size * (s_100 - s_110),
             1./X.size * (s_101 - s_111)]
        p_00_bounds = (0, 1)
        p_01_bounds = (0, 1) # Pr(\tilde{Y}=1 | A=0, Y_hat=1)
        p_10_bounds = (0, 1)
        p_11_bounds = (0, 1)
        A_eq = [[float(s_000+s_010)/(s_000+s_001), float(s_001+s_011)/(s_000+s_001),-float(s_100+s_110)/(s_100+s_101), -float(s_101+s_111)/(s_100+s_101) ],
                [float(s_000+s_010)/(s_010+s_011), float(s_001+s_011)/(s_010+s_011), -float(s_100+s_110)/(s_110+s_111),-float(s_101+s_111)/(s_110+s_111)]]
        b_eq = [0, 0]

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=((0,1),(0,1),(0,1),(0,1)),options={"disp":True})
        theta = res.x
        print (theta)
        #plt.plot(theta[0], theta[1], '*', markersize=18, label='Equal-odds')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig('roc_gender.png')

        corrected_rf = corrected_model(rf,theta)
        sf = SliceFinder(corrected_rf, (X, y))
        metric = metrics.log_loss
        #metric = metrics.accuracy_score
        metrics_all = sf.evaluate_model((X, y),metric=metric)
        reference = (np.mean(metrics_all), np.std(metrics_all), len(metrics_all))
        metrics_male = sf.evaluate_model((X[X["Sex"] == 1], y[X["Sex"] == 1]), metric=metric)
        print ("all, log_loss:", np.mean(metrics_all), 'eff size:', effect_size(metrics_all, reference))
        print ("gender=Male, log_loss:", np.mean(metrics_male), 'eff size:', effect_size(metrics_male, reference))
            
        plt.figure()
        ax = plt.gca()
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        labels = ['Female', 'Male'] 
        for i in range(len(labels)):
            y_pred = corrected_rf.predict_proba(X[X['Sex'] == i].as_matrix())[:,1]
            fpr, tpr, _ = roc_curve(y[X['Sex'] == i].as_matrix().ravel(), y_pred.ravel())
            plt.plot(fpr, tpr, lw=2,label=labels[i])
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig('roc_gender_c.png')

    def test_clustering_example(self):
        sf = SliceFinder(lr, (X,y))
        metrics_all = sf.evaluate_model((X, y))
        reference = (np.mean(metrics_all), np.std(metrics_all), len(metrics_all))

        # Scale features
        scaler = StandardScaler()
        numeric_cols = ["Capital Gain", "Age", "fnlwgt", "Education-Num", "Capital Loss"]
        X_ = copy.deepcopy(X)
        X_[numeric_cols] = scaler.fit_transform(X_[numeric_cols])

        #y_pred = lr.predict(X) 
        #X_mis, y_mis = X_[y != y_pred], y[y != y_pred]
        #reduced_data_train = PCA(n_components=2).fit_transform(X_mis)
        reduced_data_train = PCA(n_components=2).fit_transform(X_)

        kmeans = KMeans(init='k-means++', n_clusters=20, n_init=10)
        kmeans.fit(reduced_data_train)

        print ('Cluster centroids')
        print ('==============================')
        print (kmeans.cluster_centers_)
        cluster_eg = kmeans.cluster_centers_[9]
        print ('==============================')
        
        x_min, x_max = reduced_data_train[:,0].min() - 1, reduced_data_train[:,0].max() + 1
        y_min, y_max = reduced_data_train[:,1].min() - 1, reduced_data_train[:,1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max-x_min)/10000), np.arange(y_min, y_max, (y_max-y_min)/10000))
        print ('figure axes')
        print ('==============================')
        print (xx.min(), xx.max(), yy.min(), yy.max())
        print ('==============================')

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        Z_ = kmeans.predict(reduced_data_train)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   alpha=0.2,
                   aspect='auto', origin='lower')

        #plt.plot(reduced_data_train[:, 0], reduced_data_train[:, 1], 'k.', markersize=2)
        plt.scatter(reduced_data_train[:, 0], reduced_data_train[:,1], c=Z_, s=30, cmap=plt.cm.Paired)
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:,0], centers[:,1], c='black',s=100,alpha=0.5)
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        #plt.scatter(centroids[:, 0], centroids[:, 1],
        #plt.scatter(centroids[9, 0], centroids[9, 1],
        #            marker='x', s=150, linewidths=2,
        #            color='b', zorder=10)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())

        cluster_id = 10
        #plt.text(cluster_eg[0]-5, cluster_eg[1]+3, 'ID=%s'%cluster_id, color='black', size=18)
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

        temp = adult_data[adult_data.columns.difference(["Target"])]
        temp = temp[Z_ == cluster_id]
        with open('cluster.txt', 'w') as f:
            for i in temp.index:
                f.write(temp.ix[i].to_string() + '\n')

        

        plt.tight_layout()
        plt.savefig('clusters.png')

        y_predicted = kmeans.predict(reduced_data_train) 
        X_cluster = X[np.array(y_predicted) == cluster_id]
        y_cluster = y[np.array(y_predicted) == cluster_id]
        metric_examples = sf.evaluate_model((X_cluster, y_cluster))
        print('avg_metric_all: %s'%np.mean(sf.evaluate_model((X,y))))
        eff_size_ = effect_size(metric_examples, reference)
        print('id: %s, size: %s, eff_size: %s, avg_metric: %s'%(cluster_id, len(X_cluster), eff_size_, np.mean(metric_examples)))
        X_avg = np.mean(X_cluster, axis=0)
        y_avg = np.mean(y_cluster, axis=0)
        X_std = np.std(X_cluster, axis=0)
        y_std = np.std(y_cluster, axis=0)
        print('X_avg: %s\n X_std: %s\n y_avg: %s\n y_std: %s'%(X_avg, X_std, y_avg, y_std))

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

        for n_clusters in range(20, 21):
            kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
            kmeans.fit(reduced_data_train) # train with mis-classified data
            kmeans.fit(reduced_data_test)
            y_predicted = kmeans.predict(reduced_data_test) # classify all data

            # cluster the original data with the leanred centroids
            # analyze each cluster (e.g., compute effect size)
            print('==========================')
            print('n_clusters: %s'%n_clusters)
            print('==========================')
            for cluster_id in np.unique(kmeans.labels_):
                X_cluster = X[np.array(y_predicted) == cluster_id]
                y_cluster = y[np.array(y_predicted) == cluster_id]
                metric_examples = sf.evaluate_model((X_cluster, y_cluster))
                eff_size_ = effect_size(metric_examples, reference)
                print('id: %s, size: %s, eff_size: %s, avg_metric: %s'%
                        (cluster_id, len(X_cluster), eff_size_, np.mean(metric_examples)))
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
            print ('---------------------\nmetric: %s'%(s.metric))
            print ('size: %s'%(s.size))

    def test_parallel_find_slice(self):
        df = adult_data.sample(frac=0.2)
        X_, y_ = df[df.columns.difference(["Target"])], df["Target"]
        sf = SliceFinder(lr, (X_, y_))
        recommendations = sf.find_slice(k=5, degree=3, max_workers=8)
        
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
            print ('---------------------\nmetric: %s'%(s.metric))
            print ('size: %s'%(s.size))

    def test_scalability(self):
        import time
        dataset = 'UCI'
        """
        main_df = pd.read_csv("data/creditcard.csv")
        main_df = main_df.dropna()
        main_df.head()
        feature_size = len(main_df.columns)
        class_index = feature_size -1 

        fraud_raw = main_df[main_df['Class'] == 1]
        normal_raw = main_df[main_df['Class'] == 0]

        # Undersample the normal transactions
        percentage = len(fraud_raw)/float(len(normal_raw))
        normal = normal_raw.sample(frac=percentage)
        fraud = fraud_raw
        cc_data = fraud.append(normal)

        X, y = cc_data[cc_data.columns.difference(["Class"])], cc_data["Class"]

        reg_model = RandomForestClassifier(criterion='entropy',n_estimators=100)
        lr = reg_model.fit(X,y)
        """
        k = 10
        sf = SliceFinder(lr, (X,y))
        start_time = time.time()
        recommendations = sf.find_slice(k=k, epsilon=0.4, degree=6)
        end_time = time.time()
        full_time = end_time - start_time

        decisions = []
        for x_, y_ in zip(X.as_matrix(), y.as_matrix()):
            y_p = lr.predict([x_])
            if y_p == y_:
                decisions.append(0)
            else:
                decisions.append(1)
        
        clf = DecisionTree((X, y), lr)
        start_time = time.time()
        clf = clf.fit(max_depth=6, min_size=100)
        recommendations_dt = clf.recommend_slices(k=k, min_effect_size=0.3)
        end_time = time.time()
        full_time_dt = end_time - start_time

        times, t_err = list(), list()
        times_dt, tdt_err = list(), list()
        recalls, r_err = list(), list()
        recalls_dt, rdt_err = list(), list()

        for r in np.array(range(1,10,2))/10.:
            time_ = list()
            time_dt = list()
            recall_ = list()
            recall_dt = list()
            for rep in range(0,3):
                #df = adult_data.sample(frac=r)
                df = cc_data.sample(frac=r)
                #X_, y_ = df[df.columns.difference(["Target"])], df["Target"]
                X_, y_ = df[df.columns.difference(["Class"])], df["Class"]

                sf = SliceFinder(lr, (X_, y_))
             
                '''
                decisions = []
                for x__, y__ in zip(X_.as_matrix(), y_.as_matrix()):
                    y_p = lr.predict([x__])
                    if y_p == y__:
                        decisions.append(0)
                    else:
                        decisions.append(1)
                '''
                clf = DecisionTree((X_, y_), lr)

                # sample size vs run time (single core)
                start_time = time.time()
                recs = sf.find_slice(k=k, epsilon=0.4, degree=6)
                end_time = time.time()            
                time_.append(end_time-start_time)

                start_time = time.time()
                clf = clf.fit(max_depth=6, min_size=10)
                recs_dt = clf.recommend_slices(k=k, min_effect_size=0.4)
                end_time = time.time()
                time_dt.append(end_time-start_time)

                # sample size vs top-10 ranking
                correct = 0.
                for s in recommendations:
                    for s_ in recs:
                        if s_.__str__() == s.__str__():
                            correct += 1.
                            break
                recall_.append(correct / k)
                correct = 0.
                for n in recommendations_dt:
                    n_desc = ' '.join(n.__ancestry__() + [n.__str__().split(',')[0]])
                    for n_ in recs_dt:
                        n_desc_ = ' '.join(n_.__ancestry__() + [n_.__str__().split(',')[0]])
                        if n_desc == n_desc_:
                            correct += 1.
                            break

                recall_dt.append(correct / k)
            times.append(np.mean(time_))
            t_err.append(np.std(time_))
            times_dt.append(np.mean(time_dt))
            tdt_err.append(np.std(time_dt))
            recalls.append(np.mean(recall_))
            r_err.append(np.std(recall_))
            recalls_dt.append(np.mean(recall_dt))
            rdt_err.append(np.std(recall_dt))
        times.append(full_time)
        recalls.append(1.)
        times_dt.append(full_time_dt)
        recalls_dt.append(1.)
        t_err.append(0.)
        r_err.append(0.)
        tdt_err.append(0.)
        rdt_err.append(0.)

        with open('scale_%s.p'%dataset,'wb') as handle:
            pickle.dump([times,times_dt,recalls,recalls_dt,t_err,tdt_err,r_err,rdt_err], handle)
        
        stats = []
        with open('scale_%s.p'%dataset,'rb') as handle:
            stats = pickle.load(handle)
        print (stats)
        print (stats[4])
        sampling_frac = list(np.array(range(1,10,2))/10.) + [1.0]

        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams.update({'font.size': 17})

        plt.figure(3, figsize=(12,5))
        plt.subplot(1,2,1)
        plt.errorbar(sampling_frac, stats[0], yerr=stats[4] , fmt='o-', label='LS', linewidth=3.0, markersize=10)
        plt.errorbar(sampling_frac, stats[1], yerr=stats[5], fmt='v-', label='DT', linewidth=3.0, markersize=10)
        plt.legend()
        plt.xlabel('Sampling Fraction', fontsize=17)
        plt.ylabel('Wall-clock Time (sec)', fontsize=17)
        plt.subplot(1,2,2)
        plt.errorbar(sampling_frac, stats[2], yerr=stats[6] , fmt='o-', label='LS', linewidth=3.0, markersize=10)
        plt.errorbar(sampling_frac, stats[3], yerr=stats[7], fmt='v-', label='DT', linewidth=3.0, markersize=10)
        plt.legend()
        plt.xlabel('Sampling Fraction', fontsize=17)
        plt.ylabel('Recall', fontsize=17)
        plt.ylim([0,1.05])
        plt.tight_layout()
        plt.savefig('%s_scalability.pdf'%(dataset))

        plt.figure(4, figsize=(12,5))
        width = 0.1
        plt.subplot(1,2,1)
        plt.bar(sampling_frac, stats[0], width, edgecolor='black',  label='LS', yerr=stats[4])
        plt.bar(sampling_frac, stats[1], width, edgecolor='black', label='DT', yerr=stats[5])
        plt.legend()
        plt.xlabel('Sampling Fraction', fontsize=17)
        plt.ylabel('Wall-clock Time (sec)', fontsize=17)
        plt.subplot(1,2,2)
        plt.bar(sampling_frac, stats[2], width, edgecolor='black', label='LS', yerr=stats[6])
        plt.bar(sampling_frac, stats[3], width, edgecolor='black', label='DT', yerr=stats[7])
        plt.legend()
        plt.xlabel('Sampling Fraction', fontsize=17)
        plt.ylabel('Recall', fontsize=17)
        plt.ylim([0,1.05])
        plt.tight_layout()
        plt.savefig('%s_scalability2.pdf'%(dataset))
        
    def test_scalability2(self):
        import time
        dataset = 'Census'
        """
        k = 10

        times, t_err = list(), list()
        recalls, r_err = list(), list()

        for n_thread in range(1, 5, 1):
            time_ = list()
            recall_ = list()
            for rep in range(0,1):
                df = adult_data
                X_, y_ = df[df.columns.difference(["Target"])], df["Target"]

                sf = SliceFinder(lr, (X_, y_))
             
                # sample size vs run time (single core)
                start_time = time.time()
                recs = sf.find_slice(k=k, epsilon=0.4, degree=6, max_workers=n_thread)
                end_time = time.time()            
                time_.append(end_time-start_time)

            times.append(np.mean(time_))
            t_err.append(np.std(time_))

        with open('parallel_scale_%s.p'%dataset,'wb') as handle:
            pickle.dump([times,t_err], handle)
        """
        stats = []
        with open('parallel_scale_%s.p'%dataset,'rb') as handle:
            stats = pickle.load(handle)
       
        n_threads = range(1,5,1)

        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams.update({'font.size': 17})

        plt.figure(3, figsize=(6,5))
        plt.errorbar(n_threads, stats[0], yerr=stats[1] , fmt='o-', label='LS', linewidth=3.0, markersize=10)
        plt.legend()
        plt.xlabel('Number of Workers', fontsize=17)
        plt.ylabel('Wall-clock Time (sec)', fontsize=17)
        plt.tight_layout()
        plt.savefig('%s_scalability_parallel.pdf'%(dataset))

if __name__ == '__main__':
    unittest.main()
