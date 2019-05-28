# coding=utf-8
from models import *
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt
from joblib import dump, load

df = pd.read_csv('dota2Dataset/dota2Test.csv', header=None)


y_test = df.iloc[:, 0].to_numpy()
x_test = df.iloc[:, 1:].to_numpy()

#print(x_test)
#print(y_test)

df = pd.read_csv('dota2Dataset/dota2Train.csv', header=None)
y_train = df.iloc[:, 0].to_numpy()
x_train = df.iloc[:, 1:].to_numpy()
#logic_reg(x_train, y_train, x_test, y_test)
clf = load('logic_reg.model')
judge(clf, x_test, y_test)


# start = time.time()
# clf = svm.SVC()
# clf.fit(x_train, y_train)
# result = clf.predict(x_test)
# print(clf.score(x_test, y_test))
# print(time.time() - start)




