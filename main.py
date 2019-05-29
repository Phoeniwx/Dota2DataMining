# coding=utf-8
from models import *
import pandas as pd
from Tsne import *
# import numpy as np
# from sklearn import svm
# from sklearn.manifold import TSNE
# import time
# import matplotlib.pyplot as plt
# from joblib import dump, load

df = pd.read_csv('dota2Dataset/dota2Test.csv', header=None)


y_test = df.iloc[:, 0].to_numpy()
x_test = df.iloc[:, 1:].to_numpy()

df = pd.read_csv('dota2Dataset/dota2Train.csv', header=None)
y_train = df.iloc[:, 0].to_numpy()
x_train = df.iloc[:, 1:].to_numpy()
# logic_reg(x_train, y_train, x_test, y_test)
clf = load('models/random_forest.model')
judge(clf, x_test, y_test)

# svm_(x_train, y_train, x_test, y_test)
# linear_svm(x_train, y_train, x_test, y_test)
# naive_bay(x_train, y_train, x_test, y_test)
# decision_tree(x_train, y_train, x_test, y_test)
# random_forest(x_train, y_train, x_test, y_test)
# t_sne(x_train, y_train)


# start = time.time()
# clf = svm.SVC()
# clf.fit(x_train, y_train)
# result = clf.predict(x_test)
# print(clf.score(x_test, y_test))
# print(time.time() - start)




