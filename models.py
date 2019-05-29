# coding=utf-8
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from joblib import dump, load
import time
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn import metrics


def plot_roc(clf, test_x, test_y):
    y_score = clf.decision_function(test_x)
    fpr, tpr, _ = roc_curve(test_y, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic Curve')
    plt.legend(loc="lower right")

    plt.show()


# 评估算法，混淆矩阵+ROC
def judge(clf, test_x, test_y):
    predict_y = clf.predict(test_x)
    accuracy = accuracy_score(test_y, predict_y)
    precision = metrics.precision_score(test_y, predict_y, average='macro', pos_label=1)
    recall = metrics.recall_score(test_y, predict_y, average='macro', pos_label=1)
    print('accuracy: {}, precision: {}, recall: {}'.format(accuracy, precision, recall))

    matrix = confusion_matrix(test_y, predict_y)
    classes = unique_labels(test_y, predict_y)
    print(matrix)

    title = 'Confusion matrix'
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(matrix.shape[1]),
           yticks=np.arange(matrix.shape[0]),
           title=title,
           # ... and label them with the respective list entries
           xticklabels=classes,
           yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')
    fmt = 'd'
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()


# SVM
def svm_(train_x, train_y, test_x, test_y):
    print('----SVM----')
    start = time.time()
    clf = SVC(C=0.25)
    clf.fit(train_x, train_y)
    print("Train time: %s" % (time.time() - start))
    dump(clf, 'SVM.model')
    # judge(clf, test_x, test_y)


# Linear SVM
def linear_svm(train_x, train_y, test_x, test_y):
    print('----linear_SVM----')
    start = time.time()
    clf = SVC(kernel='linear', C=0.1, gamma=0.001)
    clf.fit(train_x, train_y)
    print("Train time: %s" % (time.time() - start))
    dump(clf, 'linearSVM.model')
    # judge(clf, test_x, test_y)


# 逻辑回归
def logic_reg(train_x, train_y, test_x, test_y):
    print('----Logistic Regression----')
    start = time.time()
    clf = LogisticRegression(solver='sag')
    clf.fit(train_x, train_y)
    print("Train time: %s" % (time.time() - start))
    dump(clf, 'logic_reg.model')
    judge(clf, test_x, test_y)


# 朴素贝叶斯
def naive_bay(train_x, train_y, test_x, test_y):
    print('----Bayes----')
    start = time.time()
    clf = GaussianNB()
    clf.fit(train_x, train_y)
    print("Train time: %s" % (time.time() - start))
    dump(clf, 'GaussianNB.model')
    # judge(clf, test_x, test_y)


# KNN
def knn(train_x, train_y, test_x, test_y):
    print('----KNN----')
    start = time.time()
    clf = KNeighborsClassifier()
    clf.fit(train_x, train_y)
    print("Train time: %s" % (time.time() - start))
    dump(clf, 'KNN.model')
    # judge(clf, test_x, test_y)


# 决策树
def decision_tree(train_x, train_y, test_x, test_y):
    print('----Decision Tree----')
    start = time.time()
    clf = DecisionTreeClassifier()
    clf.fit(train_x, train_y)
    print("Train time: %s" % (time.time() - start))
    dump(clf, 'decision_tree.model')
    # judge(clf, test_x, test_y)


# 随机森林
def random_forest(train_x, train_y, test_x, test_y):
    print('----Random Forest----')
    start = time.time()
    clf = RandomForestClassifier()
    clf.fit(train_x, train_y)
    print("Train time: %s" % (time.time() - start))
    dump(clf, 'random_forest.model')
    # judge(clf, test_x, test_y)
