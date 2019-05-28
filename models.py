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
import time


def judge(clf, test_x, test_y):
    predict_y = clf.predict(test_x)
    cm = confusion_matrix(test_y, predict_y)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
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
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def svm(train_x, train_y, test_x, test_y):
    start = time.time()
    clf = SVC(C=0.25)
    clf.fit(train_x, train_y)
    print("Train time: %s" % (time.time() - start))
    judge(clf, test_x, test_y)


def linear_svm(train_x, train_y, test_x, test_y):
    start = time.time()
    clf = SVC(kernel='linear', C=0.1, gamma=0.001)
    clf.fit(train_x, train_y)
    print("Train time: %s" % (time.time() - start))
    judge(clf, test_x, test_y)


def logic_reg(train_x, train_y, test_x, test_y):
    print('----Logistic Regression----')
    start = time.time()
    clf = LogisticRegression(solver='lbfgs')
    clf.fit(train_x, train_y)
    print("Train time: %s" % (time.time() - start))
    judge(clf, test_x, test_y)


def naive_bay(train_x, train_y, test_x, test_y):
    start = time.time()
    clf = GaussianNB()
    clf.fit(train_x, train_y)
    print("Train time: %s" % (time.time() - start))
    judge(clf, test_x, test_y)


def knn(train_x, train_y, test_x, test_y):
    start = time.time()
    clf = KNeighborsClassifier()
    clf.fit(train_x, train_y)
    print("Train time: %s" % (time.time() - start))
    judge(clf, test_x, test_y)


def decision_tree(train_x, train_y, test_x, test_y):
    start = time.time()
    clf = DecisionTreeClassifier()
    clf.fit(train_x, train_y)
    print("Train time: %s" % (time.time() - start))
    judge(clf, test_x, test_y)


def random_forest(train_x, train_y, test_x, test_y):
    start = time.time()
    clf = RandomForestClassifier()
    clf.fit(train_x, train_y)
    print("Train time: %s" % (time.time() - start))
    judge(clf, test_x, test_y)