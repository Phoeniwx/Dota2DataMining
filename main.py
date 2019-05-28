# coding=utf-8
from models import *
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt

df = pd.read_csv('dota2Dataset/dota2Test.csv', header=None)


y_test = df.iloc[:, 0].to_numpy()
x_test = df.iloc[:, 1:].to_numpy()

#print(x_test)
#print(y_test)

df = pd.read_csv('dota2Dataset/dota2Train.csv', header=None)
y_train = df.iloc[:, 0].to_numpy()
x_train = df.iloc[:, 1:].to_numpy()
logic_reg(x_train,y_train,x_test,y_test)

# X_tsne = TSNE(n_components=2, random_state=33).fit_transform(x_train)
# start = time.time()
# clf = svm.SVC()
# clf.fit(x_train, y_train)
# result = clf.predict(x_test)
# print(clf.score(x_test, y_test))
# print(time.time() - start)
# font = {"color": "darkred",
#         "size": 13,
#         "family" : "serif"}
# plt.style.use("dark_background")
# plt.figure(figsize=(8.5, 4))
# plt.subplot(1, 2, 1)
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train, alpha=0.6,
#             cmap=plt.cm.get_cmap('rainbow', 10))
# plt.title("t-SNE", fontdict=font)
# cbar = plt.colorbar(ticks=range(10))
# cbar.set_label(label='digit value', fontdict=font)
# plt.clim(-0.5, 9.5)
# plt.subplot(1, 2, 2)

