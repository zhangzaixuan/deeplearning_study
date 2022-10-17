import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

"""
大多数的数据分布都是非线性回归，线性函数没办法比较好的拟合数据分布，或者使用线性回归最小似然估计的偏差过大；
"""
x = np.arange(6).reshape(3, 2)
print(x)
# degree,多项式次数，多项式中最高阶的次数，interaction_only，true,包含偏置项，多项式是否含有常数系数，include_bias true 只找交互作用的多项式矩阵；
# PolynomialFeatures degree=2, *, interaction_only=False, include_bias=True,
poly = PolynomialFeatures(2)
print(poly.fit_transform(x))
# 样本数据
x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
Y = y.reshape(-1, 1)
poly = PolynomialFeatures(degree=2)
poly.fit(X)
X2 = poly.transform(X)
lr = LinearRegression()
lr.fit(X2, Y)
# 多项式回归
y_predict = lr.predict(X2)
plt.scatter(x, y)
plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')
plt.savefig('./preprocessing_img/polynomial.png')
plt.pause(5)
lr.fit(X, Y)
# 经典线性回归
class_y_predict = lr.predict(X)
plt.plot(x, class_y_predict, color='r')
plt.scatter(x, y)
plt.savefig('./preprocessing_img/linear.png')
plt.pause(5)


