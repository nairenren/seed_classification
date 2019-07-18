# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:05:35 2019

@author: PC - 33
"""

from sklearn import tree, svm, neighbors
from sklearn.metrics import accuracy_score
#DT----------------------------------------------------------------------------
def decisionTreeClassify(x_train, x_test, y_train, y_test):
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(x_train,y_train)
    predictions = classifier.predict(x_test)
    
    return round(accuracy_score(y_test,predictions) * 100, 2)
#SVM---------------------------------------------------------------------------
def svmClassify(x_train, x_test, y_train, y_test, kernel):
    classifier = svm.SVC(gamma='scale', kernel=kernel)
    classifier.fit(x_train,y_train)
    predictions = classifier.predict(x_test)

    return round(accuracy_score(y_test,predictions) * 100, 2)
#KNN---------------------------------------------------------------------------
def knnClassify(x_train, x_test, y_train, y_test, k):
    classifier = neighbors.KNeighborsClassifier(n_neighbors=k)
    classifier.fit(x_train,y_train)
    predictions = classifier.predict(x_test)

    return round(accuracy_score(y_test,predictions) * 100, 2)
#Load--------------------------------------------------------------------------
import numpy as np
import timeit
dataset = np.loadtxt("seeds_dataset.txt")
x = dataset[:,0:7]
y = dataset[:,7]
#Split-------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.5)
k = [3, 5, 7]#random?
kernel = ['rbf','linear','poly']#sigmoid, precomputed
#Classify----------------------------------------------------------------------
for i in range(3):
    print("\nIteration - {}".format(i+1))
    #DT========================================================================
    start_time = timeit.default_timer()
    print("Decision Tree: ", decisionTreeClassify(x_train, x_test, y_train, y_test), "%")
    elapsed = timeit.default_timer() - start_time
    print(">Execution time(in seconds): {0:.5f}".format(elapsed))
    #SVM=======================================================================
    start_time = timeit.default_timer()
    print("SVM Kernel({}): ".format(kernel[i]), svmClassify(x_train, x_test, y_train, y_test, kernel[i]), "%")
    elapsed = timeit.default_timer() - start_time
    print(">Execution time(in seconds): {0:.5f}".format(elapsed))
    #KNN=======================================================================
    start_time = timeit.default_timer()
    print("KNN k-({}): ".format(k[i]), knnClassify(x_train, x_test, y_train, y_test, k[i]), "%")
    elapsed = timeit.default_timer() - start_time
    print(">Execution time(in seconds): {0:.5f}".format(elapsed))