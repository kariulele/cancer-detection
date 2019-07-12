#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ImageLoader import ImgStorage
from featureExtractor import make_train_and_test_list, image_to_KAZEfeature, image_to_ORBfeature,image_to_SIFTfeature


from sklearn.gaussian_process.kernels import RBF
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis




from visu import *
import matplotlib.pyplot as plt

storage = ImgStorage('../images/', '../gt_img.csv')


    
train_image, train_x, train_y, test_image, test_x, test_y = make_train_and_test_list(storage, 0.2, image_to_SIFTfeature)


def test_classify(clf,name):
    clf.fit(train_x, train_y)
    score = clf.score(test_x,test_y)    
    print(name,"'s score = ",score)
    
    
classifiers = [
    KNeighborsClassifier(10),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=10, n_estimators=100, max_features=10),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

for i in range(len(classifiers)):
    test_classify(classifiers[i],names[i])
    

#show_test_res(clf, test_image,test_x, test_y)