# coding=utf-8
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# 数据降维可视化
def t_sne(x, y):
    clf = TSNE(n_components=2, random_state=33)
    X_tsne = clf.fit_transform(x)
    font = {"color": "darkred",
            "size": 13,
            "family": "serif"}
    plt.style.use("dark_background")
    plt.figure(figsize=(8.5, 4))
    # plt.subplot(1, 2, 1)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, alpha=0.6,
                cmap=plt.cm.get_cmap('rainbow', 10))
    plt.title("t-SNE", fontdict=font)
    color_bar = plt.colorbar(ticks=range(10))
    color_bar.set_label(label='digit value', fontdict=font)
    plt.clim(-0.5, 9.5)
    plt.show()
