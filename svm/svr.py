import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'STFangsong']
plt.rcParams['axes.unicode_minus'] = False
N = 30
np.random.seed(0)
x = np.sort(np.random.uniform(0, 2 * np.pi, N), axis=0)
y = 2 * np.sin(x) + 0.2 * np.random.randn(N)
x = x.reshape(-1, 1)
"""
001 linear:线性核函数  适用线性问题。
002 poly:多项式核函数    试用偏线性问题。
003 rbf:高斯核函数（默认） 适用非线性问题。
004 sigmoid:双曲正切核函数  适用偏非线性问题。
"""
# 径向基函数 radial-basis-function（其实就是高斯核函数）
svr_rbf = svm.SVR(kernel='rbf', gamma=0.2, C=100)
svr_rbf.fit(x, y)
print(svr_rbf.score(x, y))
svr_line = svm.SVR(kernel='linear', C=100)
svr_line.fit(x, y)
print(svr_line.score(x, y))
# polynomial 多项式
svr_poly = svm.SVR(kernel='poly', degree=3, C=100)
svr_poly.fit(x, y)
print(svr_poly.score(x, y))


x_test = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_rbf = svr_rbf.predict(x_test)
y_linear = svr_line.predict(x_test)
y_ploy = svr_poly.predict(x_test)
plt.scatter(x, y, c='b')
plt.show()

plt.figure(figsize=(8, 10), facecolor='w')
plt.plot(x_test, y_rbf, 'r-', linewidth=2, label='高斯核函数')
plt.plot(x_test, y_linear, 'g-', linewidth=2, label='线性核函数')
plt.plot(x_test, y_ploy, 'b-', linewidth=2, label='多项式核函数')
plt.plot(x, y, 'mo', markersize=6, markeredgecolor='k')

plt.scatter(x[svr_rbf.support_], y[svr_rbf.support_], s=200, c='r', marker='*', edgecolors='k', label='支持向量机',
            zorder=10)
plt.legend(loc='lower left', fontsize=12)
plt.title('支持向量机回归：svr model', fontsize=15)
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(b=True, ls=':')
# plt.tight_layout(2) 高版不接受入参了
plt.tight_layout()
# plt.show()
plt.savefig('./out_data/image/svr.png')
plt.pause(5)

