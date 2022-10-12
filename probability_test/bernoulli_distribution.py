from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np
# 'SimHei' windows 中文标签 'Songti SC', 'STFangsong' mac 宋体和仿宋字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'STFangsong']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False

"""
伯努利分布是典型的两点分布（离散随机变量分布）
变量x有两个取值 0，1
对应概率为 p,1-p
伯努利的期望为 E(x)=1*p+0*(1-p)=p
E(x²)=1²*p+0²*(1-p)=p
方差为D(x)=E(x²)-(E(x))²=p-p²=p(1-p)
"""
n = 10
p = 0.50
k = np.arange(0, 10)
binomail = binom.pmf(k, n, p)
plt.plot(k, binomail)
plt.title('binomail 伯努利分布：n=%i,p=%0.2f' % (n, p), fontsize=15)
plt.xlabel('随机变量 number of success x')
plt.ylabel('概率值 proaibility of success p')
# plt.show()
plt.savefig('./binomail.png')
plt.pause(1)
