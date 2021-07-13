#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 19:13:45 2021

@author: davidmasip
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import preprocessing


def run_classifiers(input_file, variables):
    Data_CAC_P = pd.read_excel(input_file, sheet_name='CACS>400')
    Data_CAC_N = pd.read_excel(input_file, sheet_name='CACS<400')

    n_vars = len(variables)
    positive=[]
    negative=[]
    for var in variables:
        positive.append([Data_CAC_P[var].to_numpy()])
        negative.append([Data_CAC_N[var].to_numpy()])

    P = np.vstack(positive)
    N = np.vstack(negative)

    n_P = P.shape[1]
    n_N = N.shape[1]

    all_data = np.concatenate((P.T, N.T))

    y = np.hstack([[0] * n_P, [1] * n_N])

    n_methods = 9
    n_cv = 10
    n_classes = 2

    names = ['Logistic Regression', 'KNN', 'SVM', 'Gaussian Process', 'Decision tree', 'Random Forest', 'AdaBoost',
             'Quadratic Classifier', 'Naive Bayes']

    scores = np.zeros((n_methods, n_cv))

    i = 0
    cv = StratifiedKFold(n_splits=n_cv)
    for train_ind, test_ind in cv.split(all_data, y):
        method = 0

        #np.bincount(y[train_ind]), np.bincount(y[test_ind])))
        train, test, y_train, y_test = all_data[train_ind], all_data[test_ind], y[train_ind], y[test_ind]

        scaler = preprocessing.StandardScaler().fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)

        # Logistic regression
        clf_log = LogisticRegression(random_state=0, multi_class='auto').fit(train, y_train)
        scores[method, i] = clf_log.score(test, y_test)
        method = method + 1

        # Knn

        clf_knn = KNeighborsClassifier(n_neighbors=3)
        clf_knn.fit(train, y_train)
        scores[method, i] = clf_knn.score(test, y_test)
        method = method + 1

        # SVM

        clf_svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf_svm.fit(train, y_train)
        scores[method, i] = clf_svm.score(test, y_test)
        method = method + 1

        # Gaussian process
        clf_GP = GaussianProcessClassifier(1.0 * RBF(1.0))
        clf_GP.fit(train, y_train)
        scores[method, i] = clf_GP.score(test, y_test)
        method = method + 1

        # Decision tree
        clf_tree = DecisionTreeClassifier(max_depth=5)
        clf_tree.fit(train, y_train)
        scores[method, i] = clf_tree.score(test, y_test)
        method = method + 1

        # Random forest
        clf_RF = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        clf_RF.fit(train, y_train)
        scores[method, i] = clf_RF.score(test, y_test)
        method = method + 1

        # Adaboost
        clf_adaboost = AdaBoostClassifier()
        clf_adaboost.fit(train, y_train)
        scores[method, i] = clf_adaboost.score(test, y_test)
        method = method + 1

        # Quadratic classifier
        clf_QC = QuadraticDiscriminantAnalysis()
        clf_QC.fit(train, y_train)
        scores[method, i] = clf_QC.score(test, y_test)
        method = method + 1

        # NB
        clf_NB = GaussianNB()
        clf_NB.fit(train, y_train)
        scores[method, i] = clf_NB.score(test, y_test)
        method = method + 1

        i = i + 1

    # Results
    results = scores.mean(axis=1)
    for i, method in enumerate(names):
        # print(f'{results[i]}')
        print("{:.3f}".format(results[i]))
    print('---------------')
    print("{:.3f}".format(np.average(results)))
    # print(f'Mean classifiers: {np.average(results)}')
    print('---------------')
    return results, np.average(results)

