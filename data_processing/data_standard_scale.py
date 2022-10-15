import numpy as np
from sklearn import preprocessing

"""
zscore 标准化对x1,x2....,xn进行变换，获取yᵢ
yᵢ=(xᵢ-mean(x))/sᵢs=sqrt((1/(n-1))*\sum_{i=1}^{n}(x-\overline{x}))
"""

x_ori = np.array([[1, -1, 3], [3, 0, 1], [0, 1, -1]])
"""
zscore 标准函数 preprocessing.scale
x_mean 均值
x_std  标准差
"""
x_scale = preprocessing.scale(x_ori)
x_mean = x_scale.mean(axis=0)
x_std = x_scale.std(axis=0)
# print(x_scale.var(axis=0))
print(x_scale, x_mean, x_std)

# 标准化是对列进行处理
x_fit = preprocessing.StandardScaler().fit(x_ori)
x_transform = x_fit.transform(x_ori)
x_new = [[1, -1, 3]]
print(x_fit.transform(x_new))
