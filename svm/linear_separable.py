import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.datasets._samples_generator import make_blobs

"""
支持向量机的基础是线性可分，直观来看，在二维场景下，用一条直线将分类样本点切分开，在三维场景下，可以构建一个平面（有可能是曲平面）将空间点切分开
"""
sbn.set()
X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
x_fit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
for i, j in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(x_fit, i * x_fit + j, '-k')
plt.xlim(-1, 3.5)
plt.savefig('./out_data/image/linear_separable.png')
plt.pause(5)

for i, j, k in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    y_fit = i * x_fit + j
    plt.plot(x_fit, y_fit, '-k')
    plt.fill_between(x_fit, y_fit - k, y_fit + k, edgecolor='none', color='#AAAAAA', alpha=0.4)
plt.xlim(-1, 3.5)
plt.savefig('./out_data/image/linear_width_separable.png')
plt.pause(5)
# 线性可分推导
"""
svm模型会让数据样本点距离超平面尽可能的远：所有数据样本点在各自分类向量的两边；
超平面方程为 kx+b=0,数据样本点和超平面的距离为1，可以得到两个边界函数，kx+b=-1,kx+b=1
将函数形式转换成向量模式 kᵀx+b=0,则边界函数为：kᵀx+b=-1，kᵀx+b=1，进一步转换可以获得（kᵀx1+b）-（kᵀx2+b）=2（x1,x2 位于不同的两个边界中）
最后可获得x1,x2 到超平面的距离为d1=d2=kᵀ(x1-x2)/2||w||₂=2/2||w||₂=1/||w||₂,两个边界的距离为2/||w||₂
"""
# 核函数
"""
线性可分并不是总能达成的，数据总会不尽如人意，所以才需要去驯服
如果线性不可分，可以尝试一下曲线可分，比如圆周方程 x²+y²=4,就可以将圆周线内外的点进行二分类；
处理线性不可分问题，可以将其转换成一个基于映射函数的线性问题，x²+y²=4，a+b=4,a=x²,b=y²(a=x²,b=y²作为映射函数)
在二维空间,x=(x1,x2),不存在一条直线将数据样本点划分开，通过核函数z=\varnothing[x1²,x2²],核函数本质是一个输入空间到特征空间的映射函数

svm算法中，存在线性核函数，多项式核函数，高斯核函数，线性核函数 K(x,z)=x·z+c(K,kernel,c 为惩罚因子),多项式核函数 K(x,z)=(x·z+c)ᵖ(p次多项式函数)

高斯核函数K(x,z)=e^(-||x-z||²)/2(\sigma)²

在生产使用svm方法时，如果特征和样本数量相差不大，使用线性和多项式函数，特征和样本数量相差比较大，选择高斯核函数，
当特征和样本有数量级的差异时，无脑高斯核函数。
"""