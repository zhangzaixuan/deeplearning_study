import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets._samples_generator import make_blobs
from sklearn.decomposition import PCA
"""
降纬算法用于降低数据纬度，防止纬度爆炸，本身是算力不足，特征纬度的线性增加，通常伴随计算量指数级的增加
"""
# 样本数10000，特征数3个，聚簇4个
X, y = make_blobs(n_samples=10000, n_features=3, centers=[[6, 6, 6], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
                  cluster_std=[0.2, 0.1, 0.2, 0.2], random_state=9)
fig = plt.figure()
rect_tuple = tuple([0., 0., 1., 1.])
# ax = Axes3D(fig, rect=[i for i in [0., 0., 1., 1.]], elev=30, azim=20)
# 搞不懂这里为什么要强校验类型，只是warining,能work
ax = Axes3D(fig, rect=tuple([0., 0., 1., 1.]), elev=30, azim=20)
plt.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o')
# plt.show()
plt.savefig('./preprocessing_img/pca_ori.png')
plt.pause(5)

# print(type([i for i in [0., 0., 1., 1.]]))
# print(type([[0., 0., 1., 1.]]))
# print(type([0., 0., 1., 1.]))

pca = PCA(n_components=2)
pca.fit(X)
x_new = pca.fit_transform(X)
plt.scatter(X[:, 0], X[:, 1], marker='o')
# plt.show()
plt.savefig('./preprocessing_img/pca_components.png')
plt.pause(5)
