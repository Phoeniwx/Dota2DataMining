# coding=utf-8
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# 数据降维可视化
def t_sne(x, y):
    print('----T-sne----')
    # clf = TSNE(n_components=2, random_state=0)
    # X_tsne = clf.fit_transform(x)
    # np.save('tsne.npy', X_tsne)
    X_tsne = np.load('tsne.npy')
    color = ['b', 'r', 'orange']
    colors = [color[i+1] for i in y]
    font = {"color": "darkred",
            "size": 13,
            "family": "serif"}
    #plt.style.use("dark_background")
    plt.figure(figsize=(8.5, 4))
    # plt.subplot(1, 2, 1)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, alpha=0.6, s=1)
    plt.title("t-SNE")
    # color_bar = plt.colorbar(ticks=range(-1, 2))
    # color_bar.set_label(label='digit value', fontdict=font)
    plt.clim(-0.5, 9.5)
    # plt.savefig("t_sne.jpg")
    plt.show()
